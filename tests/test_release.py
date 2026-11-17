"""Synthetic CPU-only regression tests; no real experiments or downloads."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from maxn_code.analyze_predictions import (METRICS, analyze_bundle, binary_metrics,
                                           group_indices, summarize, validate_bundle)
from maxn_code.evaluate_checkpoint import collect_c16, collect_c17, export, load_model
from maxn_code.maxn_camelyon16 import AuxiliaryYAdaptiveTopK, MaxNMilSingleBranch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def c16_fixture(folder):
    features = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.],
                         [.9, 0., 1., 0.], [0., .9, 0., 1.]], dtype=np.float32)
    with h5py.File(folder / "val_patch_feat.h5", "w") as handle:
        handle.create_dataset("dataset_1", data=features)
    arrays = dict(patch_label=[1, 0, 1, 0], patch_corresponding_slide_label=[1, 0, 1, 0],
                  patch_corresponding_slide_index=[1, 0, 1, 0],
                  patch_corresponding_slide_name=["test_001_pos", "test_002_neg", "test_001_pos", "test_002_neg"])
    for name, value in arrays.items():
        np.save(folder / ("val_" + name + ".npy"), np.asarray(value))
    np.save(folder / "coords.npy", np.array([[0, 0], [512, 0], [0, 512], [512, 512]]))
    return features


class ModelTests(unittest.TestCase):
    def test_parameter_counts(self):
        for dimension, latent, expected in [(512, 35, 105570), (768, 35, 138338), (512, 24, 101137)]:
            model = MaxNMilSingleBranch(in_dim=dimension, instance_latent_dim=latent)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), expected)

    def test_two_gammas_are_different(self):
        scores = torch.tensor([.9, .8] + [.1] * 14).unsqueeze(-1)
        pool = AuxiliaryYAdaptiveTopK(train_pooling="adaptive_topk", adaptive_gamma=.5, topk_gamma=1.)
        value, count = pool._pool_scores(scores)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(value.item(), .85, places=6)
        pool.topk_gamma = .5
        value, count = pool._pool_scores(scores)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(value.item(), .875, places=6)

    def test_inference_remains_max(self):
        pool = AuxiliaryYAdaptiveTopK(train_pooling="mean", eval_pooling="max", topk_gamma=.5)
        pool.eval()
        value, count = pool._pool_scores(torch.tensor([[.9], [.2]]))
        self.assertEqual(count, 1)
        self.assertAlmostEqual(value.item(), .9, places=6)

    def test_mean_component(self):
        pool = AuxiliaryYAdaptiveTopK(train_pooling="mean", eval_pooling="max")
        value, count = pool._pool_scores(torch.tensor([[.9], [.3]]))
        self.assertEqual(count, 2)
        self.assertAlmostEqual(value.item(), .6, places=6)

    def test_anchor_uses_actual_count(self):
        model = MaxNMilSingleBranch(in_dim=2, instance_latent_dim=2)
        probabilities = torch.tensor([[.95], [.1], [.8]])
        logits = torch.logit(probabilities)
        features = torch.tensor([[1., 0.], [.98, .1], [0., 1.]])
        loss = model._anchor_expansion_loss(logits, probabilities, torch.zeros(3, dtype=torch.long), torch.ones(1), features)
        expected = F.binary_cross_entropy_with_logits(logits[:2, 0], torch.ones(2))
        torch.testing.assert_close(loss, expected)

    def test_anchor_never_supervises_negative_bag(self):
        model = MaxNMilSingleBranch(in_dim=2, instance_latent_dim=2)
        probabilities = torch.tensor([[.99], [.99]])
        loss = model._anchor_expansion_loss(torch.logit(probabilities), probabilities,
                torch.zeros(2, dtype=torch.long), torch.zeros(1), torch.ones(2, 2))
        self.assertEqual(loss.item(), 0.)

    def test_prediction_is_deterministic_without_rng_draws(self):
        model = MaxNMilSingleBranch(in_dim=4, instance_latent_dim=3, eval_pooling="max").eval()
        features = torch.ones(5, 4)
        ids = torch.zeros(5, dtype=torch.long)
        state = torch.get_rng_state().clone()
        first = model.predict_bag_score(features, ids)
        second = model.predict_bag_score(features, ids)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


class DataAndExportTests(unittest.TestCase):
    def test_checkpoint_dimensions_and_interleaved_c16_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            features = c16_fixture(folder)
            model = MaxNMilSingleBranch(in_dim=4, instance_latent_dim=3, eval_pooling="max").eval()
            checkpoint = folder / "weights.pth"
            torch.save(model.state_dict(), checkpoint)
            loaded, dimension, latent = load_model(checkpoint, torch.device("cpu"))
            self.assertEqual((dimension, latent), (4, 3))
            bundle = collect_c16(loaded, str(folder), torch.device("cpu"), dimension, folder / "coords.npy")
            with torch.no_grad():
                mean, _ = model.encoder(torch.from_numpy(features))
                direct = torch.sigmoid(model.aux_y.fc_ins(mean)).squeeze(-1).numpy()
            np.testing.assert_array_equal(bundle["patch_scores"], direct)
            output = folder / "evaluation"
            metrics = export(bundle, output, "MaxN-MIL", 6, "c16", dimension, latent, checkpoint)
            self.assertIn("patch_auprc", metrics)
            with np.load(output / "predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["patch_scores"], direct)
                validate_bundle(saved)
            csv = np.loadtxt(output / "froc_csv/test_001_pos.csv", delimiter=",")
            np.testing.assert_array_equal(csv[:, 1:], np.array([[0, 0], [0, 512]]))

    def test_bad_coordinate_alignment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            c16_fixture(folder)
            np.save(folder / "wrong.npy", np.zeros((3, 2)))
            model = MaxNMilSingleBranch(in_dim=4, instance_latent_dim=3, eval_pooling="max").eval()
            with self.assertRaises(ValueError):
                collect_c16(model, str(folder), torch.device("cpu"), 4, folder / "wrong.npy")

    def test_c17_center_split(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            lines = ["slide_id,center"]
            with h5py.File(folder / "features.h5", "w") as handle:
                for center in range(5):
                    for label in range(2):
                        name = "center%d_slide%d" % (center, label)
                        lines.append("%s,%d" % (name, center))
                        slide = handle.create_group(name)
                        slide.attrs["label"] = label
                        slide.create_dataset("feat", data=np.ones((2, 4), dtype=np.float32))
                        slide.create_dataset("coords", data=np.zeros((2, 2), dtype=np.int64))
            (folder / "metadata.csv").write_text("\n".join(lines) + "\n")
            model = MaxNMilSingleBranch(in_dim=4, instance_latent_dim=3, eval_pooling="max").eval()
            result = collect_c17(model, str(folder / "features.h5"), str(folder / "metadata.csv"), torch.device("cpu"), 4, 2021)
            self.assertEqual(len(result["bag_scores"]), 4)
            self.assertTrue(all(name.startswith(("center3", "center4")) for name in result["slide_names"]))
            self.assertNotIn("patch_labels", result)


class AnalysisTests(unittest.TestCase):
    def bundle(self):
        ids = np.repeat(np.arange(6), 1000)
        labels = np.zeros(len(ids), dtype=int)
        labels[:1] = 1
        labels[1000:1010] = 1
        labels[2000:2100] = 1
        return dict(bag_labels=np.array([1, 1, 1, 0, 0, 0]),
                    bag_scores=np.array([.6, .7, .8, .2, .3, .4]),
                    slide_names=np.array(["test_001_pos", "test_002_pos", "test_003_pos", "test_004_neg", "test_021_neg", "test_114_neg"]),
                    patch_labels=labels, patch_scores=np.where(labels, .8, .1), patch_slide_index=ids,
                    slide_threshold=np.array(.5), patch_threshold=np.array(.5))

    def test_burden_boundaries_and_shared_negatives(self):
        bundle = self.bundle()
        groups = group_indices(bundle)
        self.assertEqual(groups["small"], [3, 0])
        self.assertEqual(groups["medium"], [3, 1])
        self.assertEqual(groups["large"], [3, 2])
        self.assertEqual(len(groups["all_test"]), 6)
        self.assertEqual(len(groups["all_retained"]), 4)

    def test_thresholds_remain_frozen_and_froc_is_not_invented(self):
        records = analyze_bundle(self.bundle(), "MaxN", 6)
        self.assertTrue(all(r["slide_threshold"] == .5 and r["patch_threshold"] == .5 for r in records))
        self.assertTrue(all(r["froc"] is None for r in records))

    def test_missing_tumor_masks_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze_bundle(self.bundle(), "MaxN", 6, mask_cache={})

    def test_constant_scores_keep_original_youden_behavior(self):
        result = binary_metrics(np.array([0, 1]), np.array([.5, .5]))
        self.assertTrue(np.isposinf(result["threshold"]))
        self.assertEqual(result["acc"], .5)

    def test_average_rank_and_sample_sd(self):
        records = []
        for name, values in [("A", [.7, .9]), ("B", [.8, .8]), ("C", [.6, .6])]:
            for seed, value in enumerate(values):
                records.append(dict(method=name, seed=seed, group="small", **{m: value for m in METRICS}))
        summary = {r["method"]: r for r in summarize(records)}
        self.assertEqual(summary["A"]["average_rank"], 1.5)
        self.assertEqual(summary["B"]["average_rank"], 1.5)
        self.assertAlmostEqual(summary["A"]["slide_auc_sd"], np.std([.7, .9], ddof=1))
        self.assertEqual(summary["C"]["rank_metric_count"], 6)

    def test_incomplete_seed_pairs_are_rejected(self):
        records = [dict(method=name, seed=seed, group="small", **{m: .8 for m in METRICS})
                   for name, seed in [("A", 1), ("B", 2)]]
        with self.assertRaises(ValueError):
            summarize(records)

    def test_single_run_sd_is_unavailable(self):
        record = dict(method="A", seed=6, group="small", **{m: .8 for m in METRICS})
        summary = summarize([record])[0]
        self.assertTrue(all(summary[m + "_sd"] is None for m in METRICS))

    def test_offline_cli_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            inputs = []
            for method in ["A", "B"]:
                for seed in [6, 7]:
                    path = folder / ("%s_%d.npz" % (method, seed))
                    np.savez_compressed(path, **self.bundle(), method=method, seed=seed, feature_dim=4)
                    inputs.append(str(path))
            output = folder / "analysis"
            command = [sys.executable, "-m", "maxn_code.analyze_predictions", "--inputs", *inputs,
                       "--output", str(output)]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = (output / "analysis.json").read_bytes()
            report = json.loads(saved)
            self.assertEqual(len(report["records"]), 20)
            self.assertFalse(report["froc_computed"])
            with (output / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["rank_metric_count"] == "5" for row in rows))
            self.assertTrue(all(float(row["average_rank"]) == 1.5 for row in rows))
            self.assertTrue(all(row["froc_mean"] == "" for row in rows))
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(saved, (output / "analysis.json").read_bytes())


class FROCTests(unittest.TestCase):
    def setUp(self):
        try:
            from maxn_code import froc
        except ModuleNotFoundError as exc:
            if exc.name in ["tifffile", "skimage"]:
                self.skipTest("Optional FROC dependencies are not installed")
            raise
        self.froc = froc

    def test_case_subset_and_center_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            np.savetxt(folder / "test_001_pos.csv", [[.9, 0, 0], [.2, 512, 0]], delimiter=",")
            np.savetxt(folder / "test_002_neg.csv", [[.1, 0, 0]], delimiter=",")
            mask = np.zeros((64, 64), dtype=int)
            mask[8, 8] = 1
            cache = {"test_001": (mask, [])}
            result = self.froc.evaluate_seed(str(folder), cache, 90, 200, 0, 256, 0, (), "center", 512, .25)
            self.assertEqual(result["n_slides"], 2)
            self.assertEqual(result["n_tumors"], 1)
            self.assertAlmostEqual(result["froc"], 1.)
            selected = self.froc.evaluate_seed(str(folder), cache, 90, 200, 0, 256, 0, (), "center", 512, .25,
                                              include_slides={"test_001"})
            self.assertEqual(selected["n_slides"], 1)

    def test_missing_positive_mask_is_not_negative(self):
        with tempfile.TemporaryDirectory() as temp:
            np.savetxt(Path(temp) / "test_001_pos.csv", [[.9, 0, 0]], delimiter=",")
            with self.assertRaises(ValueError):
                self.froc.evaluate_seed(temp, {}, 90, 200, 0, 256, 0, (), "center", 512, .25)

    def test_single_seed_uncertainty_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "froc"
            argv = ["froc", "--csv_root", temp, "--mask_dir", temp,
                    "--seeds", "6", "--out_dir", str(output)]
            result = dict(froc=.4, sens=np.full(6, .4), n_slides=2,
                          n_tumors=1, total_in=10, total_kept=10)
            with patch.object(sys, "argv", argv), patch.object(self.froc, "build_mask_cache", return_value={}), \
                    patch.object(self.froc, "evaluate_seed", return_value=result):
                self.froc.main()
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertIsNone(metadata["sample_sd"])
            self.assertEqual(metadata["ci95"], [None, None])


class RecipeAndCLITests(unittest.TestCase):
    def test_runner_refuses_existing_output_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            command = [sys.executable, "scripts/run_experiment.py", "c16_r18",
                       "--data", "/synthetic/missing", "--output", temp]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_recipes_and_variants(self):
        spec = importlib.util.spec_from_file_location("recipes", ROOT / "scripts/run_experiment.py")
        recipes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recipes)
        for recipe in recipes.RECIPES:
            for variant in recipes.VARIANTS:
                args = SimpleNamespace(recipe=recipe, variant=variant, data="/synthetic/features", csv="/synthetic/metadata.csv",
                                       output="/synthetic/output", seeds=None, similarity=.75)
                command = recipes.build_command(args)
                options = dict(zip(command[3::2], command[4::2]))
                self.assertEqual(options["--adaptive_gamma"], "0.5")
                self.assertEqual(options["--topk_gamma"], "1.0")
                self.assertEqual(options["--eval_pooling"], "max")
                self.assertEqual(options["--early_stop_patience"], "0" if recipe == "c17_r18" else "10")

    def test_package_entry_points_and_direct_c17_help(self):
        commands = [[sys.executable, "-m", name, "--help"] for name in
                    ["maxn_code.maxn_camelyon16", "maxn_code.maxn_camelyon17",
                     "maxn_code.evaluate_checkpoint", "maxn_code.analyze_predictions"]]
        commands.append([sys.executable, "maxn_code/maxn_camelyon17.py", "--help"])
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
