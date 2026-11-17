# MaxN-MIL

Code for **Max and Neighbors: Similarity-Guided Max-Pooling Multiple Instance Learning for Whole Slide Image Classification**.

MaxN-MIL expands training supervision through an adaptive top-ranked pool and feature-similar neighbors of a confident anchor. Inference uses the maximum instance probability, with the latent mean instead of random sampling.

## Release Contents

- C16 ResNet18/CTransPath and C17 training, including module switches.
- Explicit paper recipes in `scripts/run_experiment.py`.
- Frozen-checkpoint evaluation and patch CSV export.
- Archived mask-based FROC, offline tumor-burden analysis and summary statistics.
- Synthetic CPU-only regression tests.
- Current manuscript tables in `docs/main_results.csv`.

This is the classification MaxN-MIL release, not an all-baseline benchmark bundle. Baseline implementations and the exploratory industrial/survival tasks are not included. Raw slides, masks, features, coordinates, weights and private research logs are not redistributed. See [release scope](docs/RELEASE_SCOPE.md) and [protocol](docs/PROTOCOL.md).

## Environment

```bash
conda create -n maxn python=3.10
conda activate maxn
pip install -r requirements.txt
# Optional, for mask-based FROC:
pip install -r requirements-froc.txt
```

Install a PyTorch build compatible with your hardware. Evaluation defaults to CPU unless a device is specified.

## Data Format

### Camelyon16

For both the `train` and `val` prefixes, supply:

```text
<split>_patch_feat.h5                         # dataset_1: [patches, D]
<split>_patch_label.npy                       # patch GT, evaluation only
<split>_patch_corresponding_slide_label.npy   # slide label per patch
<split>_patch_corresponding_slide_index.npy   # contiguous zero-based bag ID
<split>_patch_corresponding_slide_name.npy    # slide name per patch
```

All arrays are row-aligned. `train` is the training pool, internally split 80:20. Despite its filename, `val` is the external test cohort, not the internal validation set. The feature package contains 269 training-pool bags and 128 test bags. ResNet18 and CTransPath features have 512 and 768 dimensions, respectively. Patch labels do not enter the training loss or checkpoint-selection criterion.

For FROC, obtain aligned level-0 patch top-left coordinates and official tumor masks. Coordinates cannot be inferred from patch order.

### Camelyon17

Supply an HDF5 file and metadata CSV. Each slide group contains `feat`, `coords` and a `label` attribute. The CSV contains `slide_id` and zero-based `center`. Labels are binarized as 0 versus 1/2/3. Centers 3 and 4 form the test set; the other centers are split 80:20 using the run seed. The released package used in the experiments yields 239 training, 60 validation and 200 test bags.

## Training Recipes

These commands print the configuration without training. Remove `--dry-run` to train.

```bash
python scripts/run_experiment.py c16_r18 \
  --data /path/to/c16_resnet18 --output /path/to/runs/c16_r18 --dry-run

python scripts/run_experiment.py c16_ctrans \
  --data /path/to/c16_ctranspath --output /path/to/runs/c16_ctrans --dry-run

python scripts/run_experiment.py c17_r18 \
  --data /path/to/c17_features.h5 --csv /path/to/camelyon17.csv \
  --output /path/to/runs/c17_r18 --dry-run
```

| Setting | C16 ResNet18 | C16 CTransPath | C17 ResNet18 |
|---|---:|---:|---:|
| Input / latent dimensions | 512 / 35 | 768 / 35 | 512 / 24 |
| KL / classification coefficients | 0.01 / 1 | 0.01 / 1 | 1 / 100 |
| Maximum epochs | 100 | 100 | 100 |
| Early-stop patience | 10 | 10 | Disabled |
| Checkpoint criterion | Validation slide AUC | Validation slide AUC | Validation slide AUC |
| Split seeds / runs | 6-10 / 5 | 1-30 / 30 | 2021-2025 / 5 |

Shared settings are batch size 3, AdamW learning rate and weight decay `1e-4`, scan limit 16, neighbor coefficient 0.05, cosine threshold 0.75, neighbor budget 64, anchor confidence 0.9, and max inference.

**The two gamma parameters are different.** Paper gamma is `--adaptive_gamma 0.5`, controlling the inclusion threshold. `--topk_gamma 1.0` means the selected probabilities are averaged without mixing in an additional max term. The old examples incorrectly used `--topk_gamma 0.5`.

Equivalent C16 ResNet18 command:

```bash
python -m maxn_code.maxn_camelyon16 \
  --dataset_dir /path/to/c16_resnet18 --in_dim 512 --instance_latent_dim 35 \
  --seeds 6,7,8,9,10 --kl_divergence_coef 0.01 --aux_loss_multiplier_y 1 \
  --train_pooling adaptive_topk --eval_pooling max \
  --topk_max 16 --adaptive_gamma 0.5 --topk_gamma 1.0 \
  --anchor_coef 0.05 --anchor_sim_threshold 0.75 \
  --anchor_expand_topk 64 --anchor_min_score 0.9 \
  --batch_size 3 --epochs 100 --early_stop_patience 10 \
  --checkpoint_dir /path/to/checkpoints
```

Historical C16 seeds fix the internal split but do not reset the global training RNG per run. This behavior is retained, so rerunning a split seed does not guarantee identical weights. C17 resets Python, NumPy and PyTorch RNGs per seed. C16 retains its training-loss-based learning-rate scheduler; C17 does not use that scheduler.

The original pooled-probability input to `BCEWithLogitsLoss` is retained to match archived experiments, not silently replaced with corrected BCE.

## Components and Similarity Sensitivity

Use `--variant full`, `topk_only`, `anchor_only`, `max_only` or `mean`. These switches alter only training pooling and neighbor supervision. The same-weight max-only control is not the historical FocusMIL reference, whose KL coefficient differs.

```bash
python scripts/run_experiment.py c16_r18 --variant topk_only \
  --data /path/to/c16_resnet18 --output /path/to/runs/topk --dry-run

python scripts/run_experiment.py c16_r18 --similarity 0.9 \
  --data /path/to/c16_resnet18 --output /path/to/runs/sim09 --dry-run
```

Use similarity 0.5, 0.75 or 0.9 for sensitivity; the reference remains 0.75.

## Evaluate Saved Weights

```bash
python -m maxn_code.evaluate_checkpoint c16 \
  --checkpoint /path/to/checkpoint.pth --data /path/to/c16_resnet18 \
  --method MaxN-MIL --seed 6 --output /path/to/evaluation/seed6 \
  --coords /path/to/val_patch_coords.npy

python -m maxn_code.evaluate_checkpoint c17 \
  --checkpoint /path/to/checkpoint.pth --data /path/to/c17_features.h5 \
  --csv /path/to/camelyon17.csv --method MaxN-MIL --seed 2021 \
  --output /path/to/evaluation/c17_seed2021
```

Dimensions are inferred from the checkpoint and validated against the data. Outputs are `metrics.json`, aligned `predictions.npz`, and, when C16 coordinates are supplied, a `froc_csv/` directory of per-slide `score,x,y` files. Coordinates remain top-left locations. Existing evaluation outputs are not overwritten.

Historical slide ACC/F1 and patch F1 use test-cohort Youden thresholds. This reproduces the archived development protocol, not threshold-independent held-out evaluation. AUC and AUPRC do not require a classification threshold.

## FROC and Tumor-Burden Analysis

```bash
python -m maxn_code.froc \
  --csv_root /path/to/evaluation --seed_prefix seed --seeds 6,7,8,9,10 \
  --mask_dir /path/to/official_masks --out_dir /path/to/froc_results

python -m maxn_code.analyze_predictions \
  --inputs /path/to/evaluation/seed*/predictions.npz \
  --mask_dir /path/to/official_masks --output /path/to/burden_results
```

The FROC command accepts per-slide CSVs either directly inside each seed directory or in its `froc_csv/` subdirectory. Defaults match the archived recipe: level 5, fixed level-0 resolution 0.243 um/pixel, patch-center hits, 90 um tolerance, 200 um ITC threshold, center offset 256 pixels, no NMS and no candidate cap. Sensitivity is averaged at 0.25, 0.5, 1, 2, 4 and 8 FP/slide. This is the archived Camelyon-style evaluator, not a claim of an untouched official challenge scorer.

Burden groups use GT-positive patch proportions in positive slides: small below 1%, medium from 1% to below 10%, large at least 10%. The groups share negative controls. Historical exclusions `test_021` and `test_114` are retained for FROC and burden strata, without relabeling. Global test thresholds stay fixed across strata.

Without masks and exported CSVs, FROC is explicitly unavailable. Offline analysis also writes sample-SD summaries and Average Rank. All included methods must use matching cohorts, seeds and available metrics for a rank comparison.

## Current Manuscript Results

MaxN-MIL means and sample standard deviations:

| Setting | Slide AUC | Slide ACC | Slide F1 | Patch AUPRC | Patch F1 | FROC |
|---|---:|---:|---:|---:|---:|---:|
| C16 ResNet18 | 0.8151 +/- 0.0205 | 0.8422 +/- 0.0116 | 0.7503 +/- 0.0219 | 0.4478 +/- 0.0168 | 0.4353 +/- 0.0122 | 0.3882 +/- 0.0086 |
| C16 CTransPath | 0.9593 +/- 0.0143 | 0.9437 +/- 0.0183 | 0.9240 +/- 0.0249 | 0.6633 +/- 0.0247 | 0.6475 +/- 0.0376 | 0.6612 +/- 0.0438 |
| C17 ResNet18 | 0.8735 +/- 0.0077 | 0.8830 +/- 0.0057 | 0.8306 +/- 0.0096 | - | - | - |

All baseline rows are in [main_results.csv](docs/main_results.csv), including the current attention-backbone C17 MHIM result. These are archived manuscript summaries, not new scores from this release check. Old PathGen and extra C17 ablation tables are no longer presented as current paper benchmarks; their research records are preserved.

## Checks

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

Tests use synthetic CPU data, do not train on Camelyon, and do not start GPU jobs.

Before sharing a release, run the identity scan with a private list of names, handles, email addresses and affiliations kept outside this repository:

```bash
python scripts/check_anonymity.py --terms-file /path/to/private-terms.txt
```

The scan covers release files, not Git commit metadata or the external anonymous mirror. Share the anonymous view or a history-free source archive, not a clone of the identified source repository. Third-party reference credits are retained.

## Citation

Citation will be added after publication.
