# Phase-boundary evaluation validation — 2026-08-27

## Automated checks

- `python -m compileall -q src Phase_Boundary_Training_Local_Fallback.py tests`: passed.
- `python -m unittest discover -s tests -v`: 12/12 tests passed.
- Colab notebook JSON and Python-cell syntax validation: passed for all 11 cells after replacing Colab magics with no-op statements.
- `git diff --check`: passed.

The tests cover cluster-ID permutation invariance, matched confusion matrices, boundary metrics, seed stability, confidence intervals, CSV/image ground-truth loading, label collapsing, spatial/sample alignment rejection, repeated-seed artifact saving, and the rule that fallback K-means outputs must never be labeled DEC/IDEC.

## Committed local benchmark

Run: `Local_Fallback_Runs/run_kmeans_benchmark_20260827_194059`

| Benchmark | Seeds | Matched accuracy | ARI | NMI | Silhouette | Boundary F1 (1 px) | Exact boundary IoU | Mean boundary distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian blobs | 5 | 0.978571 | 0.936817 | 0.902366 | 0.587576 | N/A | N/A | N/A |
| Spatial curved phases | 5 | 0.989583 | 0.968215 | 0.944327 | 0.566923 | 0.986784 | 0.758333 | 0.143330 px |

For both synthetic benchmarks, all five seeds converged to the same partition (pairwise ARI and NMI = 1.0), and silhouette, Davies-Bouldin, and Calinski-Harabasz each selected `k=3` from `k=2..7`.

## Scope

This validation does not include a new DEC/IDEC research-data result. TensorFlow and the original H5/DM3 data are not present in the local test environment, and the repository still lacks an aligned machine-readable ground-truth phase mask. The saved run is explicitly a K-means evaluation-pipeline benchmark, not evidence of real vPCF phase recovery.
