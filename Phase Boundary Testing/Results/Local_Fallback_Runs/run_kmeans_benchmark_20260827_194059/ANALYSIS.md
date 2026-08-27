# Local K-means benchmark analysis

This run is a deterministic pipeline/evaluation benchmark. It does not train DEC or IDEC and does not use the research H5/DM3 data.

## Gaussian Blob Smoke Test

- Samples/features: 420 / 2
- Mean matched accuracy: 0.978571
- Mean ARI: 0.936817
- Mean NMI: 0.902366
- Mean silhouette: 0.587576
- Internal-metric k choices: {'scope': 'synthetic geometry diagnostic', 'best_silhouette_k': 3, 'best_davies_bouldin_k': 3, 'best_calinski_harabasz_k': 3, 'warning': 'These internal criteria do not establish semantic phase correctness.'}

## Spatial Phase Boundary Test

- Samples/features: 3072 / 6
- Mean matched accuracy: 0.989583
- Mean ARI: 0.968215
- Mean NMI: 0.944327
- Mean silhouette: 0.566923
- Mean boundary F1 (1 px tolerance): 0.986784
- Mean exact boundary IoU: 0.758333
- Mean symmetric boundary distance: 0.143330 px
- Internal-metric k choices: {'scope': 'synthetic geometry diagnostic', 'best_silhouette_k': 3, 'best_davies_bouldin_k': 3, 'best_calinski_harabasz_k': 3, 'warning': 'These internal criteria do not establish semantic phase correctness.'}

## Interpretation boundary

These results validate artifact saving, label matching, repeated-seed aggregation, cluster-count sweeps, and (for the spatial benchmark) boundary metrics. They are not evidence that DEC, IDEC, or K-means recovers phases in the real vPCF dataset.
