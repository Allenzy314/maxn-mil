# FCE-MIL

Implementation for **Feature-Consistent Evidence Expansion for Weakly Supervised Whole Slide Image Classification**.

FCE-MIL is a training-time evidence expansion strategy for regularized max-pooling MIL. It improves the instance scorer by expanding supervision from high-confidence evidence patches to feature-consistent candidate patches, while keeping the inference-time slide prediction max-pooling-based.

## Repository Contents

```text
fce_code/
  fce_camelyon16.py       # FCE-MIL training on Camelyon16 feature bags
  fce_camelyon17.py       # FCE-MIL training on Camelyon17 feature bags
  camelyon17_utils.py     # Camelyon17 split, dataloader, and evaluation helpers
requirements.txt
README.md
```

The repository contains the FCE-MIL training code used for the paper experiments. It does not redistribute raw WSIs, official annotations, pretrained feature files, checkpoints, generated heatmaps, or external FROC evaluation files.

## Environment

Create a Python environment and install dependencies:

```bash
conda create -n fce python=3.9
conda activate fce
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version if the default pip wheel is not suitable:

```bash
# Example only. Choose the command matching your CUDA version from pytorch.org.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you use the external FocusMIL/Camelyon16 FROC evaluation kit with official Camelyon masks, OpenSlide may also require the system library:

```bash
sudo apt-get install openslide-tools
```

## Data Sources and Format

The training scripts assume pre-extracted patch features. Raw WSIs and official annotations are not redistributed. In the paper experiments, Camelyon16 ResNet18 and CTransPath feature bags follow the feature package/preprocessing used by FocusMIL, while Camelyon17 ResNet18 feature bags follow the feature package/split released with AEM. Users should download the original Camelyon data and obtain or extract compatible feature bags before running the scripts.

### Camelyon16 Feature Directory

The Camelyon16 scripts expect a directory containing files such as:

```text
train_patch_feat.h5
train_patch_label.npy
train_patch_corresponding_slide_label.npy
train_patch_corresponding_slide_index.npy
train_patch_corresponding_slide_name.npy

val_patch_feat.h5
val_patch_label.npy
val_patch_corresponding_slide_label.npy
val_patch_corresponding_slide_index.npy
val_patch_corresponding_slide_name.npy
```

The HDF5 feature file should contain `dataset_1` with shape `[N_patches, D]`. The NumPy arrays are row-aligned with this feature matrix:

```text
*_patch_label.npy                       # patch-level labels, used only for evaluation/analysis
*_patch_corresponding_slide_label.npy   # slide-level label for each patch row
*_patch_corresponding_slide_index.npy   # integer slide index for each patch row
*_patch_corresponding_slide_name.npy    # slide name for each patch row
```

During training, rows with the same slide index are grouped as one MIL bag.

### Camelyon17 Feature File

The Camelyon17 scripts expect an HDF5 feature file and a metadata CSV:

```text
patch_feats_pretrain_natural_supervised.h5
camelyon17.csv
```

The HDF5 file is organized by slide ID. Each slide group should contain:

```text
/<slide_id>/feat      # [N_patches, D] patch features
/<slide_id>/coords    # [N_patches, 2] level-0 patch coordinates
attrs['label']        # original Camelyon17 label
```

The metadata CSV should include `slide_id` and `center` columns. Following the common OOD split, slides with zero-indexed `center >= 3` are used as the test set, while the remaining slides are split into training and validation sets.

## Training Examples

All paths below are placeholders and should be replaced with local paths.

### Camelyon16 FCE-MIL

```bash
python fce_code/fce_camelyon16.py \
  --dataset_dir /path/to/camelyon16_features \
  --seeds 1,2,3,4,5 \
  --train_pooling adaptive_topk \
  --eval_pooling max \
  --topk_max 16 \
  --topk_gamma 1.0 \
  --anchor_coef 0.05 \
  --anchor_sim_threshold 0.75 \
  --anchor_expand_topk 64 \
  --anchor_min_score 0.9 \
  --batch_size 3 \
  --epochs 100 \
  --checkpoint_dir checkpoints/c16_fce
```

### Camelyon17 FCE-MIL

```bash
python fce_code/fce_camelyon17.py \
  --file_path /path/to/camelyon17_features.h5 \
  --csv_path /path/to/camelyon17.csv \
  --seeds 2021,2022,2023,2024,2025 \
  --train_pooling adaptive_topk \
  --eval_pooling max \
  --topk_max 16 \
  --topk_gamma 1.0 \
  --anchor_coef 0.05 \
  --anchor_sim_threshold 0.75 \
  --anchor_expand_topk 64 \
  --anchor_min_score 0.9 \
  --batch_size 3 \
  --epochs 100 \
  --checkpoint_dir checkpoints/c17_fce
```

## FROC Evaluation

For Camelyon16 localization, FROC was computed using the FocusMIL-provided Camelyon16 FROC evaluation files together with the official Camelyon16 tumor masks. These external evaluation files are not redistributed in this repository. To reproduce the paper numbers, train FCE-MIL with the scripts above and run the external FocusMIL FROC kit under the same protocol.

Default FROC settings used by the paper:

```text
Evaluation level: 5
Level-0 resolution: 0.243 um/pixel
Detection point: patch center
Tolerance: 90 um
ITC threshold: 200 um
FROC points: 0.25, 0.5, 1, 2, 4, 8 FP/slide
```

## Main FCE-MIL Hyperparameters

```text
K_max / topk_max: 16
topk_gamma: 1.0
anchor_min_score: 0.9
anchor_sim_threshold: 0.75
anchor_expand_topk / M_max: 64
anchor_coef: 0.05
```

## Notes for Anonymous Release

Before public or review release:

- Remove local absolute paths.
- Remove checkpoints, logs, generated notebooks, generated figures, and raw outputs.
- Do not include raw WSIs or official challenge annotations.
- Do not redistribute external evaluation-kit code unless its license permits it.
- Add exact pretrained feature preparation instructions or links if redistribution is not allowed.
- Replace this note with a clean citation section after acceptance/public release.

## Citation

Citation will be added after publication.
