# Phase-boundary model evaluation

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
