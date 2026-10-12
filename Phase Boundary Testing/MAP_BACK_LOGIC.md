# Map-back logic for K=3 and K=6

The original coordinate mapping is reused. Region filling and full-image
inference are additions to that mapping, described below.

## Original mapper, unchanged

The reference is
[`VPCFToImageMapper`](../Experimentally-obtained%20vPCF%20Testing/vPCF_test_2/src/vpcf_to_image_mapper.py).
It joins a saved `sample_idx` and cluster assignment to the same sample ID in
metadata. Its `atomic_positions[0]` is original image x and
`atomic_positions[1]` is original image y. `export_cluster_coordinates` exports
those coordinates and the assigned cluster. It does not fill the pixels between
sample positions or train/predict clusters.

`src/map_h5_clusters.py` adapts the current trainer's `cluster_raw` column to
the mapper's `cluster` column, builds metadata from the original H5 file, calls
the existing mapper and plots on the original DM3. No screenshot registration,
resizing or coordinate conversion is applied. Image origin is upper left.

## Added full-image inference

The original saved H5 runs contain only 12 of 380 sample assignments. To show
every site on the image, `render_full_image_cluster_regions.py` loads the saved
K=3/K=6 DEC/IDEC final weights and predicts the other 368 samples. It does not
retrain the models or infer a sample label from its pixel position.

For each original H5 vPCF frame, it extracts the same seven statistical features
used during training. Min-max scaling is reconstructed from the first 12
training frames, then applied unchanged to all frames. Recomputing scaling over
all 380 frames would change the inputs seen by the saved models. The label is
the largest Student-t cluster probability from the saved model. All four
reloaded models exactly reproduce the original 12 saved assignments.

The script passes all 380 assignments through the original mapper, preserving
H5 frame order and atomic x/y coordinates. It saves probabilities, coordinates,
training/inference flags, fitted scaling and source-weight hashes.

## Added region visualization

`src/cluster_region_figures.py` uses a nearest-neighbor spatial lookup. For each
pixel `(x,y)` in the original 500×500 DM3:

1. Find the nearest atomic sample by Euclidean distance in original pixel units.
2. Copy that sample's predicted cluster ID into the pixel's region mask.
3. Color that mask with a discrete cluster palette and overlay it on the original
   grayscale image. Color the atomic sites with the same palette.

This is a Voronoi-style region display. It fills all image pixels, including
edges beyond the outermost samples. There is no smoothing, majority voting,
reference-label matching or assignment of physical bulk/interface identities.
The saved `cluster_regions.npz` includes the nearest sample index and distance
for each pixel, so the display can be traced to an actual source prediction.

These interpolated region edges are not measured phase boundaries. Models were
trained on 12 samples; 368 labels are inference. Raw cluster IDs are specific to
each model/run and need not have the same physical meaning across K values.

Cluster names appear only in the legend. The image itself has no `Cluster n`
text markers. The atomic-site figure gives a reference-style presentation
without filling regions.

## Generated spatial benchmark

The separate 48×64 benchmark already has one assignment per grid cell.
`src/map_synthetic_clusters.py` maps those saved sample IDs directly to their
row/column cells through the same mapper. Those region maps require no nearest-
sample filling and cannot be registered to the original synthetic atomic image.

## Saved results

- `Results/Full_Runs/synthetic_cluster_regions_20261012/k3/{dec,idec}/`
- `Results/Full_Runs/synthetic_cluster_regions_20261012/k6/{dec,idec}/`
- `Results/Deep_Clustering/run_k6_20261012/map_back_k3/`
- `Results/Deep_Clustering/run_k6_20261012/map_back_k6/`

Each original-image model folder contains `cluster_regions.png`,
`all_sample_cluster_labels.png`, the coordinate table, probabilities, region
mask and inference manifest. Earlier point-only plots are preserved.
