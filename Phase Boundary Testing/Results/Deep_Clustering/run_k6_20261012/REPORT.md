# Fixed K comparison

Same saved synthetic dataset and five seeds as the K=3 baseline. Only K changed; saved DEC/IDEC hyperparameters were frozen without retuning. Raw boundary metrics retain extra cluster edges. Ground truth contains three phases.

| Model | Mean ARI | Mean raw boundary F1 | Occupied clusters per seed |
|---|---:|---:|---|
| KMeans | 0.70607 | 0.36010 | [6, 6, 6, 6, 6] |
| DEC | 0.73016 | 0.43785 | [6, 6, 6, 6, 6] |
| IDEC | 0.69810 | 0.34402 | [6, 6, 6, 6, 6] |

Both map-back directories include all 15 final model/seed results, coordinate CSVs, raw label arrays and PNGs. The synthetic feature-channel overlay is not registered to the presentation STEM image. Six requested clusters do not imply six occupied or physically identified environments.