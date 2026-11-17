"""Evaluate archived MaxN/Focus-compatible weights without training."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from .analyze_predictions import binary_metrics, write_json
from .maxn_camelyon16 import CAMELYON_16_10x_feat, MaxNMilSingleBranch


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ["model_state_dict", "model", "state_dict"]:
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    in_dim = int(state["encoder.fc_initial.0.weight"].shape[1])
    latent = int(state["encoder.mean.weight"].shape[0])
    model = MaxNMilSingleBranch(in_dim=in_dim, instance_latent_dim=latent,
                               train_pooling="adaptive_topk", eval_pooling="max")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), in_dim, latent


@torch.no_grad()
def collect_c16(model, data, device, in_dim, coords_path=None):
    dataset = CAMELYON_16_10x_feat(data, split="val", return_bag=True)
    if dataset.all_patches.shape[1] != in_dim:
        raise ValueError("Feature dimension differs from checkpoint")
    n = dataset.num_slides
    row_ids = dataset.patch_corresponding_slide_index
    if not np.array_equal(np.unique(row_ids), np.arange(n)):
        raise ValueError("C16 bag IDs must cover 0..N-1 without gaps")
    patch_scores = np.empty(dataset.num_patches, dtype=np.float32)
    bag_scores = np.empty(n, dtype=np.float32)
    slide_names = []
    for i in range(n):
        bag, _, _, _ = dataset[i]
        indices = np.flatnonzero(row_ids == i)
        names = np.unique(dataset.patch_corresponding_slide_name[indices].astype(str))
        if len(names) != 1:
            raise ValueError("One bag contains inconsistent slide names")
        slide_names.append(str(names[0]))
        bag = bag.to(device)
        score, patches, _ = model.forward_no_sampling(bag, torch.zeros(len(bag), dtype=torch.long, device=device))
        patch_scores[indices] = patches.squeeze(-1).cpu().numpy()
        bag_scores[i] = score.item()
    bundle = dict(bag_labels=np.asarray(dataset.slide_label_all, dtype=np.int64),
                  bag_scores=bag_scores, slide_names=np.asarray(slide_names),
                  patch_scores=patch_scores, patch_labels=dataset.patch_label.astype(np.int64),
                  patch_slide_index=row_ids.astype(np.int64))
    if coords_path:
        coords = np.load(coords_path, allow_pickle=False)
        if coords.shape != (dataset.num_patches, 2) or not np.isfinite(coords).all():
            raise ValueError("Coordinate shape/values do not match the original feature rows")
        if not np.equal(coords, np.round(coords)).all():
            raise ValueError("Expected integer level-0 coordinates")
        bundle["coords"] = coords
    return bundle


@torch.no_grad()
def collect_c17(model, data, csv_path, device, in_dim, seed):
    from types import SimpleNamespace
    from .camelyon17_utils import build_HDF5_feat_dataset
    _, _, dataset = build_HDF5_feat_dataset(data, SimpleNamespace(csv_path=csv_path, seed=seed))
    scores, labels, original = [], [], []
    for item in dataset:
        bag = item["feat"].to(device)
        if bag.shape[1] != in_dim:
            raise ValueError("Feature dimension differs from checkpoint")
        score = model.predict_bag_score(bag, torch.zeros(len(bag), dtype=torch.long, device=device))
        scores.append(score.item())
        labels.append(int(item["label"]))
        original.append(int(item["label_orig"]))
    return dict(bag_scores=np.asarray(scores, dtype=np.float32), bag_labels=np.asarray(labels),
                slide_names=np.asarray(dataset.data_names), original_labels=np.asarray(original))


def export(bundle, output, method, seed, dataset, in_dim, latent, checkpoint):
    slide = binary_metrics(bundle["bag_labels"], bundle["bag_scores"])
    metrics = dict(slide_auc=slide["auc"], slide_acc=slide["acc"], slide_f1=slide["f1"],
                   slide_threshold=slide["threshold"])
    bundle.update(method=np.asarray(method), seed=np.asarray(seed), dataset=np.asarray(dataset),
                  feature_dim=np.asarray(in_dim), latent_dim=np.asarray(latent),
                  slide_threshold=np.asarray(slide["threshold"]))
    if "patch_labels" in bundle:
        from sklearn.metrics import average_precision_score
        patch = binary_metrics(bundle["patch_labels"], bundle["patch_scores"])
        metrics.update(patch_auprc=float(average_precision_score(bundle["patch_labels"], bundle["patch_scores"])),
                       patch_f1=patch["f1"], patch_auc=patch["auc"], patch_threshold=patch["threshold"])
        bundle["patch_threshold"] = np.asarray(patch["threshold"])
    if "original_labels" in bundle:
        categories = {}
        for category in [1, 2, 3]:
            selected = (bundle["original_labels"] == 0) | (bundle["original_labels"] == category)
            if len(np.unique(bundle["bag_labels"][selected])) == 2:
                categories[str(category)] = binary_metrics(bundle["bag_labels"][selected], bundle["bag_scores"][selected])
        metrics["per_category"] = categories
    output.mkdir(parents=True)
    if "coords" in bundle:
        folder = output / "froc_csv"
        folder.mkdir()
        for i, name in enumerate(bundle["slide_names"]):
            if Path(str(name)).name != str(name):
                raise ValueError("Slide names must not contain directory components")
            selected = bundle["patch_slide_index"] == i
            values = np.column_stack([bundle["patch_scores"][selected], bundle["coords"][selected]])
            np.savetxt(folder / (str(name) + ".csv"), values, delimiter=",", fmt=["%.8f", "%d", "%d"])
        bundle["froc_csv_dir"] = np.asarray(str(folder.resolve()))
    np.savez_compressed(output / "predictions.npz", **bundle)
    write_json(output / "metrics.json", dict(method=method, seed=seed, dataset=dataset,
               in_dim=in_dim, latent_dim=latent, checkpoint=str(Path(checkpoint).resolve()),
               inference="Deterministic latent mean and max instance probability",
               threshold_policy="Youden thresholds selected on the evaluated test cohort; development protocol",
               test=metrics))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["c16", "c17"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--csv")
    parser.add_argument("--coords")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", default="MaxN-MIL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists; original evaluation will not be overwritten")
    if args.dataset == "c17" and not args.csv:
        parser.error("--csv is required for C17")
    device = torch.device(args.device)
    model, in_dim, latent = load_model(args.checkpoint, device)
    if args.dataset == "c16":
        bundle = collect_c16(model, args.data, device, in_dim, args.coords)
    else:
        bundle = collect_c17(model, args.data, args.csv, device, in_dim, args.seed)
    metrics = export(bundle, output, args.method, args.seed, args.dataset, in_dim, latent, args.checkpoint)
    print("Saved evaluation:", output / "metrics.json")
    print("Slide AUC:", metrics["slide_auc"])


if __name__ == "__main__":
    main()
