"""Archived mask-based FROC protocol; requires official masks, not patch labels alone."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy.stats as st
import tifffile
from scipy import ndimage as nd
from skimage import measure


EVAL_LEVEL = 5
L0_RESOLUTION_UM_PER_PX = 0.243
EVAL_FP_POINTS = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)


def read_csv(path):
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0 or arr.shape[1] != 3 or not np.isfinite(arr).all():
        raise ValueError("Detection CSV must contain finite score,x,y rows: " + str(path))
    return arr[:, 0], arr[:, 1:3]


def infer_center_offset(coords):
    offsets = []
    for col in (0, 1):
        vals = np.unique(np.sort(coords[:, col].astype(int)))
        diffs = np.diff(vals)
        diffs = diffs[diffs > 0]
        if diffs.size:
            uniq, cnt = np.unique(diffs, return_counts=True)
            offsets.append(float(uniq[np.argmax(cnt)]) / 2.0)
    return float(np.median(offsets)) if offsets else 0.0


def spatial_nms(scores, coords, radius, max_candidates=0):
    order = np.argsort(scores)[::-1]
    if radius <= 0:
        if max_candidates > 0:
            order = order[:max_candidates]
        return order.astype(int)

    cell = float(radius)
    r2 = float(radius) * float(radius)
    grid = {}
    keep = []
    for idx in order:
        x, y = coords[idx]
        gx = int(np.floor(x / cell))
        gy = int(np.floor(y / cell))
        suppressed = False
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                for kept_idx in grid.get((nx, ny), []):
                    dx = coords[kept_idx, 0] - x
                    dy = coords[kept_idx, 1] - y
                    if dx * dx + dy * dy <= r2:
                        suppressed = True
                        break
                if suppressed:
                    break
            if suppressed:
                break
        if suppressed:
            continue
        keep.append(int(idx))
        grid.setdefault((gx, gy), []).append(int(idx))
        if max_candidates > 0 and len(keep) >= max_candidates:
            break
    return np.asarray(keep, dtype=int)


def build_mask_map(mask_dir):
    out = {}
    for p in Path(mask_dir).glob("*.tif"):
        stem = p.stem
        aliases = {stem}
        if stem.startswith("mask_"):
            aliases.add(stem[len("mask_"):])
        if stem.endswith("_mask"):
            aliases.add(stem[:-len("_mask")])
        for alias in aliases:
            out[alias] = str(p)
    return out


def read_mask_level(mask_path):
    with tifffile.TiffFile(mask_path) as tif:
        series = tif.series[0]
        levels = getattr(series, "levels", None)
        if levels is not None and len(levels) > EVAL_LEVEL:
            return levels[EVAL_LEVEL].asarray()
        return tif.pages[EVAL_LEVEL].asarray()


def compute_evaluation_mask(mask_path, tolerance_um):
    gray = read_mask_level(mask_path)
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    if gray.max() <= 2:
        tumor = gray == 2
    else:
        tumor = gray > 0
    if not np.any(tumor):
        return 0
    if tolerance_um <= 0:
        expanded = tumor
    else:
        threshold_px = tolerance_um / (L0_RESOLUTION_UM_PER_PX * (2 ** EVAL_LEVEL) * 2.0)
        expanded = nd.distance_transform_edt(~tumor) < threshold_px
    return measure.label(nd.binary_fill_holes(expanded), connectivity=2)


def compute_itc_list(em, tolerance_um, itc_um):
    if isinstance(em, int) and em == 0:
        return []
    max_label = int(np.amax(em))
    if max_label == 0:
        return []
    props = measure.regionprops(em)
    threshold = (itc_um + tolerance_um) / (L0_RESOLUTION_UM_PER_PX * (2 ** EVAL_LEVEL))
    return [i + 1 for i in range(max_label) if props[i].major_axis_length < threshold]


def _point_label(em, x, y):
    h, w = em.shape
    hx = int(x / (2 ** EVAL_LEVEL))
    hy = int(y / (2 ** EVAL_LEVEL))
    if hy < 0 or hx < 0 or hy >= h or hx >= w:
        return 0
    return int(em[hy, hx])


def _overlap_label(em, x, y, itc_set, patch_size=512.0, overlap_threshold=0.25):
    h, w = em.shape
    scale = float(2 ** EVAL_LEVEL)
    half = float(patch_size) / 2.0
    x0 = max(0, int(np.floor((x - half) / scale)))
    y0 = max(0, int(np.floor((y - half) / scale)))
    x1 = min(w, int(np.ceil((x + half) / scale)))
    y1 = min(h, int(np.ceil((y + half) / scale)))
    if x1 <= x0 or y1 <= y0:
        return 0
    roi = em[y0:y1, x0:x1]
    patch_area = float((y1 - y0) * (x1 - x0))
    valid = roi > 0
    if itc_set:
        valid &= ~np.isin(roi, list(itc_set))
    overlap = float(valid.sum()) / max(1.0, patch_area)
    if overlap < overlap_threshold:
        return 0
    labels = roi[valid].astype(int)
    if labels.size == 0:
        return 0
    counts = np.bincount(labels, minlength=int(em.max()) + 1)
    counts[0] = 0
    return int(np.argmax(counts))


def split_fp_tp(scores, coords, em, itc_labels, hit_mode="center", patch_size=512.0, overlap_threshold=0.25):
    if isinstance(em, int) and em == 0:
        return scores.tolist(), np.zeros((0,), dtype=np.float32), 0
    max_label = int(np.amax(em))
    if max_label == 0:
        return scores.tolist(), np.zeros((0,), dtype=np.float32), 0
    fp_probs = []
    tp_probs = np.zeros((max_label,), dtype=np.float32)
    itc_set = set(itc_labels)
    for score, (x, y) in zip(scores, coords):
        half = float(patch_size) / 2.0
        corner_labels = [
            _point_label(em, x - half, y - half),
            _point_label(em, x + half, y - half),
            _point_label(em, x - half, y + half),
            _point_label(em, x + half, y + half),
        ]
        if hit_mode == "overlap":
            lab = _overlap_label(
                em, x, y, itc_set, patch_size=patch_size,
                overlap_threshold=overlap_threshold,
            )
        else:
            if hit_mode == "center":
                labels = [_point_label(em, x, y)]
            elif hit_mode == "any_corner":
                labels = corner_labels
            elif hit_mode == "center_or_any_corner":
                labels = [_point_label(em, x, y)] + corner_labels
            else:
                raise ValueError(f"unknown hit_mode: {hit_mode}")
            valid_labels = [lab for lab in labels if lab > 0 and lab not in itc_set]
            lab = valid_labels[0] if valid_labels else 0
        if lab == 0 or lab in itc_set:
            fp_probs.append(float(score))
        elif score > tp_probs[lab - 1]:
            tp_probs[lab - 1] = float(score)
    return fp_probs, tp_probs, max(0, max_label - len(itc_labels))


def compute_curve(fp_lists, tp_lists, positives_per_slide):
    fp_all = np.concatenate([np.asarray(x, dtype=float) for x in fp_lists]) if fp_lists else np.array([])
    tp_all = np.concatenate([np.asarray(x, dtype=float) for x in tp_lists]) if tp_lists else np.array([])
    if fp_all.size + tp_all.size <= 1:
        return np.array([0.0]), np.array([0.0])
    fp_sorted = np.sort(fp_all)
    tp_sorted = np.sort(tp_all)
    thresholds = np.unique(np.concatenate([fp_sorted, tp_sorted]))[1:]
    idx_fp = np.searchsorted(fp_sorted, thresholds, side="left")
    idx_tp = np.searchsorted(tp_sorted, thresholds, side="left")
    n_fp = np.append(len(fp_sorted) - idx_fp, 0)
    n_tp = np.append(len(tp_sorted) - idx_tp, 0)
    avg_fp = n_fp / float(len(positives_per_slide))
    n_pos = float(sum(positives_per_slide))
    sens = n_tp / n_pos if n_pos > 0 else np.zeros_like(avg_fp)
    return avg_fp, sens


def evaluate_seed(csv_dir, mask_cache, tolerance_um, itc_um, nms_radius, center_offset, max_candidates, skip_ids, hit_mode, patch_size, overlap_threshold, include_slides=None):
    files = sorted([f for f in os.listdir(csv_dir) if f.endswith(".csv")])
    files = [f for f in files if not any(s in f for s in skip_ids)]
    if include_slides is not None:
        files = [f for f in files if canonical_slide(f[:-4]) in include_slides]
    if not files:
        raise ValueError("No detection CSVs remain for evaluation")
    fp_lists, tp_lists, positives_per_slide = [], [], []
    total_in, total_kept = 0, 0

    for fn in files:
        scores, coords = read_csv(os.path.join(csv_dir, fn))
        offset = infer_center_offset(coords) if center_offset < 0 else float(center_offset)
        coords = coords + offset
        keep = spatial_nms(scores, coords, nms_radius, max_candidates=max_candidates)
        scores = scores[keep]
        coords = coords[keep]
        total_in += len(keep) if nms_radius <= 0 else len(read_csv(os.path.join(csv_dir, fn))[0])
        total_kept += len(scores)

        base = fn[:-4]
        if base.endswith("_pos") or base.endswith("_neg"):
            base = base.rsplit("_", 1)[0]
        if fn[:-4].endswith("_pos") and base not in mask_cache:
            raise ValueError("Missing official tumor mask for " + base)
        em, itc_labels = mask_cache.get(base, (0, []))
        fp, tp, n_pos = split_fp_tp(scores, coords, em, itc_labels, hit_mode=hit_mode, patch_size=patch_size, overlap_threshold=overlap_threshold)
        fp_lists.append(fp)
        tp_lists.append(tp)
        positives_per_slide.append(n_pos)

    avg_fp, sens = compute_curve(fp_lists, tp_lists, positives_per_slide)
    eval_sens = np.interp(EVAL_FP_POINTS, avg_fp[::-1], sens[::-1])
    return {
        "froc": float(eval_sens.mean()),
        "sens": eval_sens,
        "n_slides": len(files),
        "n_tumors": int(sum(positives_per_slide)),
        "total_in": int(total_in),
        "total_kept": int(total_kept),
    }


def canonical_slide(name):
    return name[:-4] if name.endswith(("_pos", "_neg")) else name


def build_mask_cache(mask_dir, tolerance_um=90.0, itc_um=200.0):
    mask_map = build_mask_map(mask_dir)
    if not mask_map:
        raise ValueError("No TIFF masks found; FROC requires official masks")
    cache = {}
    for base, path in sorted(mask_map.items()):
        if base.startswith("test_"):
            em = compute_evaluation_mask(path, tolerance_um)
            cache[base] = (em, compute_itc_list(em, tolerance_um, itc_um))
    if not cache:
        raise ValueError("No test-slide masks found")
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_root", required=True)
    parser.add_argument("--seed_prefix", default="seed")
    parser.add_argument("--seeds", default="6,7,8,9,10")
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--tolerance_um", type=float, default=90.0)
    parser.add_argument("--itc_um", type=float, default=200.0)
    parser.add_argument("--nms_radius", type=float, default=0.0)
    parser.add_argument("--center_offset", type=float, default=256.0)
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--hit_mode", default="center", choices=["center", "any_corner", "center_or_any_corner", "overlap"])
    parser.add_argument("--patch_size", type=float, default=512.0)
    parser.add_argument("--overlap_threshold", type=float, default=0.25)
    parser.add_argument("--skip_ids", default="021,114")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("Provide at least one seed, without duplicates")
    skip_ids = tuple(s.strip() for s in args.skip_ids.split(",") if s.strip())
    Path(args.out_dir).mkdir(parents=True, exist_ok=False)

    mask_cache = build_mask_cache(args.mask_dir, args.tolerance_um, args.itc_um)
    print(
        f"[*] cached masks={len(mask_cache)}, tolerance_um={args.tolerance_um}, "
        f"itc_um={args.itc_um}, nms_radius={args.nms_radius}, "
        f"max_candidates={args.max_candidates}, hit_mode={args.hit_mode}, "
        f"patch_size={args.patch_size}, overlap_threshold={args.overlap_threshold}"
    )

    lines = []
    frocs = []
    for seed in seeds:
        csv_dir = os.path.join(args.csv_root, f"{args.seed_prefix}{seed}")
        if os.path.isdir(os.path.join(csv_dir, "froc_csv")):
            csv_dir = os.path.join(csv_dir, "froc_csv")
        res = evaluate_seed(
            csv_dir=csv_dir,
            mask_cache=mask_cache,
            tolerance_um=args.tolerance_um,
            itc_um=args.itc_um,
            nms_radius=args.nms_radius,
            center_offset=args.center_offset,
            max_candidates=args.max_candidates,
            skip_ids=skip_ids,
            hit_mode=args.hit_mode,
            patch_size=args.patch_size,
            overlap_threshold=args.overlap_threshold,
        )
        frocs.append(res["froc"])
        sens_text = ", ".join(f"{fp:g}:{s:.4f}" for fp, s in zip(EVAL_FP_POINTS, res["sens"]))
        line = (
            f"seed{seed}: FROC={res['froc']:.4f} | sens[{sens_text}] | "
            f"slides={res['n_slides']}, tumors={res['n_tumors']}, "
            f"candidates={res['total_in']}->{res['total_kept']}"
        )
        print(line)
        lines.append(line)

    frocs = np.asarray(frocs, dtype=float)
    sd = float(frocs.std(ddof=1)) if len(frocs) > 1 else None
    if sd is None:
        ci = (None, None)
        mean_line = f"mean FROC: {frocs.mean():.4f} (CI unavailable with one run)"
    else:
        ci = st.t.interval(0.95, len(frocs) - 1, loc=float(frocs.mean()), scale=st.sem(frocs)) if sd > 0 else (float(frocs.mean()), float(frocs.mean()))
        mean_line = f"mean FROC: {frocs.mean():.4f} ({ci[0]:.4f}, {ci[1]:.4f})"
    print(mean_line)
    lines.append("")
    lines.append(mean_line)
    metadata = {
        "csv_root": args.csv_root,
        "seeds": seeds,
        "tolerance_um": args.tolerance_um,
        "itc_um": args.itc_um,
        "nms_radius_level0_px": args.nms_radius,
        "center_offset": args.center_offset,
        "max_candidates": args.max_candidates,
        "hit_mode": args.hit_mode,
        "patch_size": args.patch_size,
        "overlap_threshold": args.overlap_threshold,
        "skip_ids": skip_ids,
        "frocs": frocs.tolist(),
        "mean": float(frocs.mean()),
        "sample_sd": sd,
        "ci95": [float(value) if value is not None else None for value in ci],
    }
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
