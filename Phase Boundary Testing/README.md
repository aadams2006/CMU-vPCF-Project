# Phase-boundary model evaluation

## PIGNN graph and reference-encoder development

All new physics-informed graph-model development is under [PIGNN/](PIGNN/README.md).
It includes the local peak and spatial graph inputs, reference-library encoder,
training/export scripts, tests and a cited design/evaluation rationale. It does
not change the DEC/IDEC baseline or claim validated interface recognition.

## Full original-image K=6 training on all 380 vPCFs

`run_full_sample_k6.py` trains fresh DEC/IDEC networks on every original H5
sample, using the existing seven statistical features and 7–500–500–2000–6
architecture. It fits min-max scaling on all 380 observations, pretrains for
50 epochs, then runs 200 full-batch refinement updates with targets refreshed
every five updates. Five independent seeds (42–46) are retained without selection.
This is a fixed full-image clustering protocol, with no held-out split or
generalization claim. No physical ground-truth mask is supplied.

```powershell
python run_full_sample_k6.py --output Results/Full_Runs/<new-directory>
python verify_full_sample_k6.py --run Results/Full_Runs/<new-directory>
```

Results are saved in `Results/Full_Runs/synthetic_h5_full380_k6_20261025/`:
final weights, histories, initial and final embeddings/centers, probabilities,
cluster counts, internal metrics in input and embedding spaces, seed stability,
cross-model agreement, and original-image maps for every seed. Each trained
checkpoint is reloaded into a fresh instance and checked against its saved
predictions. Source snapshots, data hashes and an artifact inventory are included.
Use `REPORT.md` for interpretation and `seed_42/{dec,idec}/cluster_regions.png`
as the fixed reference figures. Maps keep cluster names in the legend.

The older October 12 H5 runs and their 12/380 caveat remain archived in their
original folders. The new run changes training coverage, scaling and training
duration, so comparison with the smoke test is not a controlled isolation of
sample count alone. All 380 assignments in the new run are in-sample results.

## Region-style map-back figures

See [MAP_BACK_LOGIC.md](MAP_BACK_LOGIC.md) for how the existing coordinate mapper
is reused and how full-image inference and nearest-sample region visualization
extend it. Cluster names are retained in legends and removed from the images.

`Results/Full_Runs/synthetic_cluster_regions_20261012/` contains region maps
and reference-style colored atomic sites for K=3 and K=6 DEC/IDEC. All 380
original H5 samples are assigned by the saved models; models remain trained on
the original 12-frame subset. The other 368 assignments are inference, using
the original training normalization. Reloaded predictions reproduce all 12
saved assignments exactly for each of the four models.

Each model folder saves `cluster_regions.png` (original HAADF with region colors,
plus a region mask; cluster names appear only in the legend), `all_sample_cluster_labels.png`, coordinate CSVs,
probabilities, and a pixel-region NPZ with the nearest source sample and distance.
Regions extend the nearest atomic sample's assignment to neighboring pixels;
there is no smoothing or use of reference labels. These are visualization
boundaries, not measured phase boundaries. Cluster numbers are specific to each
run and do not identify a physical phase by themselves.

Reproduce with Python including TensorFlow, h5py, ncempy and the repository
dependencies: `python render_full_image_cluster_regions.py --output Results/Full_Runs/<new-region-run>`.
The reusable renderer is `src/cluster_region_figures.py`. The existing
generated-grid map-back folders now also contain `cluster_regions.png`
for every seed and model at both K values.

## K=6 rerun and K=3/K=6 map-back — 2026-10-12

The saved generated-grid benchmark was rerun with only K changed to six, using
the selected DEC/IDEC configurations and seeds 42–46. See
`Results/Deep_Clustering/run_k6_20261012/REPORT.md`. Both `map_back_k3/` and
`map_back_k6/` contain coordinate CSVs, raw cluster grids and figures for all
15 final model/seed runs. Reproduce with:

```powershell
python run_fixed_k_comparison.py --output Results/Deep_Clustering/<new-run> --k 6
```

Local source H5/DM3 files also exist under `Data/` (they are excluded from Git).
The earlier `Results/smoke_test/` used only the first 12 of 380 H5 frames,
statistical features with min-max normalization, one pretraining epoch and
two refinement iterations. That smoke-test configuration was rerun at K=6 in
`Results/Full_Runs/synthetic_h5_k6_20261012/`; it is not a full boundary study.
Original-image overlays for the existing K=3 assignments and new K=6 assignments
are in `Results/Full_Runs/synthetic_h5_map_back_k3_20261012/` and
`Results/Full_Runs/synthetic_h5_map_back_k6_20261012/`.

```powershell
../.venv/Scripts/python.exe src/train_phase_boundary_models.py --source h5 --model both --n-clusters 6 --feature-method statistical --normalize minmax --max-frames 12 --pretrain-epochs 1 --maxiter 2 --update-interval 1 --batch-size 4 --seeds 42 --output-dir Results/Full_Runs/<new-h5-run>
python src/map_h5_clusters.py --run Results/smoke_test --output Results/Full_Runs/<new-k3-map>
python src/map_h5_clusters.py --run Results/Full_Runs/<new-h5-run> --output Results/Full_Runs/<new-k6-map>
```

Both map-back scripts reuse the existing `VPCFToImageMapper`. The H5 script uses
original H5 frame IDs and atomic x/y pixel coordinates on the original DM3;
the generated-grid script maps only to its own 48×64 synthetic grid. These
datasets cannot be registered to each other. Raw cluster IDs are preserved;
bulk/interface semantics are not inferred from the presentation image.

This directory contains three deliberately separate workflows:

1. `Phase_Boundary_Training_Local_Fallback.py` is a dependency-light **K-means benchmark**. It does not train DEC or IDEC and never labels its outputs as either deep model.
2. `src/train_phase_boundary_models.py` trains and compares plain K-means, DEC, and IDEC on the same vPCF feature matrix.
3. `run_deep_benchmark.py` performs actual PyTorch DEC/IDEC training on the saved synthetic spatial benchmark, independently searches each model's hyperparameters using validation ARI, and then evaluates frozen configurations over five separate training seeds. It has no fallback training path.

## Audited deep-model benchmark

Install `requirements-deep.txt`, then run:

```bash
python run_deep_benchmark.py --output Results/deep_benchmark --trials 24
python -m unittest discover -s tests -p test_deep_benchmark.py -v
```

The output includes `REPORT.md` with every search configuration, `all_search_configurations.csv`, separate final seed metrics and confidence intervals, saved ground-truth labels and input arrays, learned embeddings, fitted centers, checkpoints, and runtime evidence of encoder/center updates. K-means is explicitly logged as center initialization only for the deep models. Raw boundary metrics preserve extra cluster edges; phase-matched metrics are also retained for baseline compatibility. The existing research-data workflow remains available for separately supplied H5/DM3 data.

The repository currently does not contain the research H5/DM3 files or an aligned machine-readable phase mask. The images under `Data/images/` are visual references; `ref-image.png` includes a legend and is not automatically used as ground truth.

## What is evaluated

Every applicable run now saves:

- internal clustering metrics: silhouette, Davies-Bouldin, and Calinski-Harabasz;
- external metrics when aligned labels exist: Hungarian-matched accuracy, macro/weighted F1, NMI, and ARI;
- spatial metrics when labels also have an aligned grid shape: tolerant boundary precision/recall/F1, exact boundary IoU/Dice, mean symmetric boundary distance, and 95th-percentile boundary distance;
- repeated-seed metrics with 95% t confidence intervals;
- pairwise seed stability and cross-model agreement;
- a repeated K-means cluster-count sweep;
- raw and matched labels, fitted centers, embeddings/features, mappings, confusion matrices, metadata, and model checkpoints.

Internal metrics describe geometric compactness and separation. They do not establish that a cluster is a physically correct phase.

## Local validation benchmark

From this directory:

```bash
python Phase_Boundary_Training_Local_Fallback.py
python -m unittest discover -s tests -v
```

The benchmark writes directly to `Results/Local_Fallback_Runs/run_kmeans_benchmark_<UTC timestamp>/` and updates `latest_run.txt` plus the run manifest. It contains both the original Gaussian-blob smoke test and a spatial curved-phase benchmark that exercises boundary metrics.

## Full research-data run

An unlabeled run must explicitly select `k`; a labeled run can infer it from the mapped labels:

```bash
python src/train_phase_boundary_models.py \
  --source h5 \
  --h5-file /path/to/vpcfs.h5 \
  --model all \
  --n-clusters 3 \
  --seeds 42 43 44 45 46 \
  --cluster-counts 2 3 4 5 6 7
```

For supervised and boundary evaluation, provide an aligned NPY/NPZ, CSV/TSV/TXT, HDF5 dataset, or label-mask image:

```bash
python src/train_phase_boundary_models.py \
  --source h5 \
  --h5-file /path/to/vpcfs.h5 \
  --ground-truth-file /path/to/phase_mask.png \
  --model all \
  --seeds 42 43 44 45 46
```

If raw labels need to be collapsed (for example, five visual categories into three documented superclasses), pass `--label-map mapping.json`. Use `--ignore-label` for an explicitly marked unknown class. The loader rejects sample-count mismatches instead of guessing registration or cropping.

The Colab notebook exposes the same model, seed, cluster-count, ground-truth, mapping, and spatial settings. Its Git branch is configured to the repository's actual default branch, `master`.

## Interpretation rules

- A local fallback result validates the evaluation and artifact pipeline only.
- A K-means result is a baseline, not DEC or IDEC.
- DEC/IDEC claims require their TensorFlow training artifacts and actual method names in metadata.
- Semantic phase claims require aligned machine-readable labels or independent physical validation.
- Phase-boundary claims require an aligned spatial grid and boundary metrics, not only sample-level clustering scores.

## Completed DEC / IDEC experiment — 2026-09-08

[Full findings and every hyperparameter configuration](Results/Deep_Clustering/run_20260908/REPORT.md) are saved with [all run artifacts](Results/Deep_Clustering/run_20260908/). The archive includes 48 search runs, 10 final deep-model runs, five K-means baseline runs, checkpoints, fitted centers, learned embeddings, synthetic labels, scores, uncertainty, plots, runtime logs, source snapshots, and an SHA-256 inventory. All 16 final tests passed. IDEC is the preferred deep model; K-means remains strongest on this synthetic benchmark. No research H5/DM3 data were used.
