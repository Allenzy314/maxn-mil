# Release Inventory

## Included

| Component | File / entry point |
|---|---|
| MaxN model, C16 data and training | `maxn_code/maxn_camelyon16.py` |
| C17 training | `maxn_code/maxn_camelyon17.py` |
| C17 split, labels, collate and metrics | `maxn_code/camelyon17_utils.py` |
| Three explicit paper recipes | `scripts/run_experiment.py` |
| Component toggles and cosine-threshold recipes | `scripts/run_experiment.py --variant / --similarity` |
| Frozen checkpoint evaluation | `python -m maxn_code.evaluate_checkpoint` |
| Coordinate-aligned detection CSV export | C16 evaluator with `--coords` |
| Mask-based FROC | `python -m maxn_code.froc` |
| Offline burden stratification, sample SD and ranks | `python -m maxn_code.analyze_predictions` |
| Current main-table numbers | `docs/main_results.csv` |
| Current burden tables | `docs/tumor_burden_results.csv` |
| Current similarity-threshold table | `docs/similarity_results.csv` |
| Current parameter counts | `docs/parameter_counts.csv` |
| Synthetic CPU regression coverage | `tests/test_release.py` |
| Pre-release identity scan | `scripts/check_anonymity.py` |

Both training scripts support direct script invocation and package invocation. The shared model is present in this repository; it is not imported from a private research directory. Runtime outputs go to user-specified directories, and evaluation refuses to overwrite an existing result directory.

## Not Included

- Raw WSIs, annotations, feature arrays, coordinate packages, pretrained encoders and trained checkpoints. Obtain these through the original data owners and feature providers.
- Complete training implementations of ABMIL, AEM, CAttMIL, TransMIL, MHIM, DTFD, Conjunctive, DSMIL and mi-Net. The main-table CSV reports archived comparisons; it is not a promise that all baselines can be trained with this release alone.
- The industrial Magnetic Tile and exploratory TCGA survival adaptations. Their appendix results are separate tasks, not applications of this classification CLI without adaptation.
- Private experiment supervisors, local logs, author information, credentials and model/data artifacts.
- The external anonymous-hosting service itself. Updating a GitHub source repository does not by itself verify that a third-party anonymous mirror has refreshed.

The old README's extra PathGen and C17 ablation tables were historical experiments, not the current manuscript's benchmark tables. They remain in repository history and the original research records; no new experiments or favorable seed selection were used to replace them.

## Reference Implementations

Use the corresponding original implementations when preparing baseline reproductions, then verify the actual architecture, feature grid, head, training criterion and patch-score convention. Sharing a model name is not enough to establish a matched configuration.

- ABMIL: <https://github.com/AMLab-Amsterdam/AttentionDeepMIL>
- AEM: <https://github.com/dazhangyu123/AEM>
- TransMIL: <https://github.com/szc19990412/TransMIL>
- MHIM-MIL: <https://github.com/DearCaat/MHIM-MIL>
- DTFD-MIL: <https://github.com/hrzhang1123/DTFD-MIL>
- DSMIL: <https://github.com/binli123/dsmil-wsi>
- mi-Net: <https://github.com/yanyongluan/MINNs>
- FocusMIL reference comparison suite: <https://github.com/Raymvp/FocusMIL-and-other-max-pooling-methods>

These links identify external sources, not a claim that their defaults produce the archived numbers. In particular, historical R18 DTFD used a different feature grid from the common-grid main methods, and the current C17 MHIM row uses an attention backbone rather than the earlier TransMIL variant. The current C16 table uses method-specific patch-score conventions. A universally max-pooled baseline re-evaluation would be a different experiment.
