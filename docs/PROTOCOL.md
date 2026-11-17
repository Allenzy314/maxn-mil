# Reproduction Protocol

## Classification Training

The paper configurations use the same three network layers before the Gaussian mean/log-variance heads and one instance classifier. C16 has latent dimension 35; C17 has dimension 24. At inference, the latent mean is used without sampling, the classifier is followed by sigmoid, and the maximum probability is the slide score. The newly added evaluation utilities do not select an alternative inference rule on the test set.

`adaptive_gamma=0.5` defines the inclusion threshold as the maximum score minus 0.5 times the population standard deviation of the top 16 scores. `topk_gamma=1.0` averages the included probabilities. It is a distinct interpolation setting, not the paper's threshold coefficient. An anchor is used only in a positive training bag if its sampled score reaches 0.9; at most 64 neighbors with raw-feature cosine similarity at least 0.75 receive the auxiliary loss. The anchor itself can be included. The default neighbor loss averages accepted neighbors across active positive bags, not a fixed denominator of 64.

For compatibility with the archived runs, the slide loss passes an already pooled probability to `BCEWithLogitsLoss`. Neighbor supervision passes raw instance logits to BCE-with-logits. The KL implementation averages across patches and latent coordinates. These conventions are preserved; this release is not the corrected-BCE, hard-mixed, diffusion, or later experimental variant.

The C16 training script uses an internally seeded 80:20 bag split but does not reset the global training RNG for each run. This is intentionally not changed. Its ReduceLROnPlateau scheduler monitors training loss. C17 resets the global RNGs per seed, uses its original center split, and has no corresponding scheduler. Configurations and split seeds do not guarantee identical scores across hardware or library versions.

The C16 full model, Top-k-only and Anchor-only recipes share KL/classification coefficients 0.01/1. The historical FocusMIL reference uses 0.001/1, so it is not a pure module-removal control at identical KL weight. The runner's max-only and mean controls deliberately retain the selected recipe's weights. They must not be labeled as reproductions of every historical reference row. C17's historical FocusMIL configurations also differ from the MaxN recipe. Current CTransPath FocusMIL/MaxN tables aggregate 30 runs, while most baseline rows aggregate five.

## Classification and Patch Metrics

Checkpoint selection uses internal validation slide AUC. Historical test ACC/F1 and patch F1 select Youden thresholds from the test scores and labels. This is a development evaluation convention, not threshold-independent held-out testing. The exported thresholds are frozen before burden stratification; the analyzer does not reselect them separately within each group. AUC and AUPRC are threshold-free.

The frozen evaluator infers dimensions from trusted state-dict checkpoints and loads them strictly. C16 patch predictions are scattered back to their original feature-row order before saving, even if rows from different bags were interleaved. Exported CSVs contain top-left coordinates, not centers. The FROC stage adds the configured center offset once.

For C16, the files prefixed `val` are the external test set. There are 128 retained test bags with 46 stored positive and 82 stored negative labels. The archived `test_021` and `test_114` cases have zero positive patch labels and conflict with their official tumor-slide status. No labels are changed by this release. Their historical exclusion leaves 46 positive and 80 negative bags in the FROC/burden cohorts. The three burden groups contain 23, 15 and 8 positive bags. A different feature package may yield different counts; the analyzer reports actual denominators rather than assuming these counts.

## Mask-Based FROC

`froc.py` retains the numerical core of the archived local evaluation wrapper. It requires official multiresolution TIFF masks with evaluation level 5. Default geometry is fixed level-0 resolution 0.243 um/pixel, 90 um tolerance, 200 um ITC threshold, center hits, 512-pixel patch size and 256-pixel center offset. NMS and candidate caps are disabled. The six operating points are 0.25, 0.5, 1, 2, 4 and 8 FP/slide. Both positive and negative slides contribute to the FP/slide denominator. The maximum detection score per evaluable connected component supplies its lesion detection score.

For exact historical compatibility, mask expansion, ITC treatment, out-of-mask detections and curve interpolation follow the archived wrapper, including its treatment of detections on excluded ITC components. This is not advertised as an untouched official blind-challenge scoring script. No alternative hit mode or tolerance is selected based on a test score in the provided recipes.

Detection CSVs use eight-decimal score serialization. The offline burden analyzer uses those exported files for FROC rather than silently substituting full-precision score arrays. Missing positive-slide masks or a mismatch between selected bags and detection files raises an error. Patch labels alone are insufficient for lesion-level FROC.

## Aggregation and Interpretation

Saved prediction bundles from multiple methods must share aligned labels, bags, feature dimensions and matching seed sets. Sample SD uses `ddof=1` and is unavailable with fewer than two runs. Average Rank uses the metric means rounded to four decimals, ranks higher values first and assigns average positions to ties. The number of metrics used for ranking is reported explicitly. Without FROC, the rank uses five metrics and must not be presented as the six-metric manuscript rank. Equal weighting of ranks does not establish statistical significance.

The public CSV tables are transcriptions of archived manuscript summaries. They are not newly trained results, and their SDs do not independently quantify uncertainty over new patients or cohorts. Differences between burden strata are descriptive; the same negative slides are shared across groups.
