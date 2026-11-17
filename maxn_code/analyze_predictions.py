"""Offline evaluation of aligned saved scores; no model or training imports."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, roc_curve

METRICS = ["slide_auc", "slide_acc", "slide_f1", "patch_auprc", "patch_f1", "froc"]
EXCLUDED = {"test_021", "test_114"}


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def slide_key(name):
    name = str(name)
    return name[:-4] if name.endswith(("_pos", "_neg")) else name


def validate_scores(labels, scores):
    if labels.ndim != 1 or labels.shape != scores.shape or labels.size == 0:
        raise ValueError("Labels and scores must be aligned, nonempty vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected probabilities in [0, 1], not logits")


def youden_threshold(labels, scores):
    validate_scores(labels, scores)
    if len(np.unique(labels)) < 2:
        return float("inf") if not labels.any() else float("-inf")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def binary_metrics(labels, scores, threshold=None):
    validate_scores(labels, scores)
    threshold = youden_threshold(labels, scores) if threshold is None else float(threshold)
    prediction = scores >= threshold
    return dict(auc=float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
                acc=float(accuracy_score(labels, prediction)),
                f1=float(f1_score(labels, prediction, zero_division=0)), threshold=threshold)


def validate_bundle(bundle):
    required = ["bag_labels", "bag_scores", "slide_names", "patch_labels", "patch_scores", "patch_slide_index"]
    if any(key not in bundle for key in required):
        raise ValueError("C16 analysis needs bag/patch scores, labels, names and patch-to-slide indices")
    validate_scores(bundle["bag_labels"], bundle["bag_scores"])
    validate_scores(bundle["patch_labels"], bundle["patch_scores"])
    n = len(bundle["bag_labels"])
    ids = bundle["patch_slide_index"]
    if ids.shape != bundle["patch_labels"].shape or not np.array_equal(np.unique(ids), np.arange(n)):
        raise ValueError("Patch-to-slide IDs must cover exactly 0..number_of_slides-1")
    if len(bundle["slide_names"]) != n or len(set(slide_key(s) for s in bundle["slide_names"])) != n:
        raise ValueError("Each bag must have a distinct canonical slide name")


def group_indices(bundle):
    validate_bundle(bundle)
    labels, ids = bundle["bag_labels"], bundle["patch_slide_index"]
    retained = [i for i, name in enumerate(bundle["slide_names"]) if slide_key(name) not in EXCLUDED]
    negatives = [i for i in retained if labels[i] == 0]
    positives = {"small": [], "medium": [], "large": []}
    for i in retained:
        if labels[i] == 1:
            fraction = float(bundle["patch_labels"][ids == i].mean())
            group = "small" if fraction < .01 else "medium" if fraction < .1 else "large"
            positives[group].append(i)
    return dict(all_test=list(range(len(labels))), all_retained=retained,
                **{group: negatives + selected for group, selected in positives.items()})


def analyze_bundle(bundle, method, seed, mask_cache=None):
    groups = group_indices(bundle)
    slide_thr = float(bundle["slide_threshold"]) if "slide_threshold" in bundle else youden_threshold(bundle["bag_labels"], bundle["bag_scores"])
    patch_thr = float(bundle["patch_threshold"]) if "patch_threshold" in bundle else youden_threshold(bundle["patch_labels"], bundle["patch_scores"])
    records = []
    if mask_cache is not None:
        missing = [slide_key(name) for i, name in enumerate(bundle["slide_names"])
                   if bundle["bag_labels"][i] and slide_key(name) not in EXCLUDED
                   and slide_key(name) not in mask_cache]
        if missing:
            raise ValueError("Missing official tumor masks for: " + ", ".join(missing))
    for group, selected in groups.items():
        if not selected:
            continue
        indices = np.asarray(selected, dtype=int)
        patch_mask = np.isin(bundle["patch_slide_index"], indices)
        slide = binary_metrics(bundle["bag_labels"][indices], bundle["bag_scores"][indices], slide_thr)
        patch = binary_metrics(bundle["patch_labels"][patch_mask], bundle["patch_scores"][patch_mask], patch_thr)
        patch_labels = bundle["patch_labels"][patch_mask]
        record = dict(method=method, seed=seed, group=group,
                      n_positive=int(bundle["bag_labels"][indices].sum()),
                      n_negative=int((bundle["bag_labels"][indices] == 0).sum()),
                      slide_auc=slide["auc"], slide_acc=slide["acc"], slide_f1=slide["f1"],
                      patch_auprc=float(average_precision_score(patch_labels, bundle["patch_scores"][patch_mask])) if patch_labels.any() else None,
                      patch_f1=patch["f1"], froc=None,
                      slide_threshold=slide_thr, patch_threshold=patch_thr)
        if mask_cache is not None:
            from .froc import evaluate_seed
            csv_dir = Path(str(bundle.get("froc_csv_dir", "")))
            if not csv_dir.is_dir() or not str(bundle.get("froc_csv_dir", "")):
                raise ValueError("FROC requires the per-slide CSV export recorded in each prediction bundle")
            result = evaluate_seed(str(csv_dir), mask_cache, 90., 200., 0., 256., 0,
                                   ("021", "114"), "center", 512., .25,
                                   include_slides={slide_key(bundle["slide_names"][i]) for i in selected})
            expected = sum(slide_key(bundle["slide_names"][i]) not in EXCLUDED for i in selected)
            if result["n_slides"] != expected:
                raise ValueError("Detection CSV cohort does not match the selected prediction cohort")
            record["froc"] = result["froc"] if result["n_tumors"] else None
        records.append(record)
    return records


def summarize(records):
    methods = sorted({r["method"] for r in records})
    groups = sorted({r["group"] for r in records})
    output = []
    for group in groups:
        subsets = {method: [r for r in records if r["method"] == method and r["group"] == group] for method in methods}
        seed_sets = [{r["seed"] for r in subset} for subset in subsets.values()]
        if any(not subset or len(subset) != len({r["seed"] for r in subset}) for subset in subsets.values()) or any(s != seed_sets[0] for s in seed_sets):
            raise ValueError("Average Rank requires unique, matching seed sets for all methods")
        available = []
        for metric in METRICS:
            flags = [r[metric] is not None for subset in subsets.values() for r in subset]
            if any(flags) and not all(flags):
                raise ValueError("Metric availability differs across methods or seeds: " + metric)
            if all(flags):
                available.append(metric)
        rows = []
        for method in methods:
            subset = subsets[method]
            row = dict(method=method, group=group, n=len(subset), rank_metric_count=len(available))
            for metric in METRICS:
                values = [r[metric] for r in subset]
                row[metric + "_mean"] = float(np.mean(values)) if metric in available else None
                row[metric + "_sd"] = float(np.std(values, ddof=1)) if metric in available and len(values) > 1 else None
            rows.append(row)
        if available:
            means = np.array([[round(r[m + "_mean"], 4) for m in available] for r in rows])
            ranks = rankdata(-means, axis=0, method="average").mean(axis=1)
            for row, rank in zip(rows, ranks):
                row["average_rank"] = float(rank)
        else:
            for row in rows:
                row["average_rank"] = None
        output.extend(rows)
    return output


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--mask_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; original analysis will not be overwritten")
    cache = None
    if args.mask_dir:
        from .froc import build_mask_cache
        cache = build_mask_cache(args.mask_dir, 90., 200.)
    records, seen, reference = [], set(), None
    for path in args.inputs:
        with np.load(path, allow_pickle=False) as data:
            bundle = {key: data[key] for key in data.files}
        method, seed = str(bundle["method"]), int(bundle["seed"])
        if (method, seed) in seen:
            raise ValueError("Duplicate method/seed prediction bundle")
        seen.add((method, seed))
        validate_bundle(bundle)
        signature = hashlib.sha256()
        for key in ["slide_names", "bag_labels", "patch_labels", "patch_slide_index"]:
            signature.update(json.dumps(json_value(bundle[key]), separators=(",", ":")).encode())
        signature.update(str(bundle.get("feature_dim", "unspecified")).encode())
        if reference is not None and reference != signature.hexdigest():
            raise ValueError("Prediction cohorts, labels, feature settings or row alignment differ")
        reference = signature.hexdigest()
        records.extend(analyze_bundle(bundle, method, seed, cache))
    summary = summarize(records)
    output.mkdir(parents=True)
    write_csv(output / "per_run.csv", records)
    write_csv(output / "summary.csv", summary)
    write_json(output / "analysis.json", dict(records=records, summary=summary,
               threshold_policy="Unstratified test-cohort Youden thresholds frozen across strata",
               excluded=sorted(EXCLUDED), froc_computed=cache is not None,
               rank_policy="Descending four-decimal metric means; ties receive average positions"))
    print(output / "summary.csv")


if __name__ == "__main__":
    main()
