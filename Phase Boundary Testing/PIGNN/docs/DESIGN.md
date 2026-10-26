# Design decisions and reasoning

## 1. Keep this work separate from DEC/IDEC

The earlier baseline compresses each raster vPCF into seven image statistics.
That is useful for an inexpensive baseline, but it discards the arrangement of
the fitted peaks. Here the graph input preserves each peak's displacement and
shape fields. The new package is independent of the previous TensorFlow classes.
Native PyTorch avoids requiring another graph framework or platform-specific
scatter extension for these small graphs. This is an implementation choice,
not a claim of superiority over other frameworks.

The method literature motivates relational processing of geometric data using
message passing [1]. Our initial encoder is an original, small implementation of
that general pattern, not a reproduction of the quantum-chemistry architecture.

## 2. Two graphs serve different scientific purposes

**Local peak graph:** one graph describes one vPCF. Nodes are fitted correlation
peaks plus an explicitly marked origin node. The origin anchors displacement
geometry and is distinguished from an observed peak. Node features are radius,
mean fitted sigma, ellipticity and an origin indicator. Raw sigma_y/sigma_x and
the unscaled coordinates remain available in the read-only source records.
Mean sigma deliberately drops directional width anisotropy in this first version.

Edges are symmetric directed six-nearest-neighbor links. Each node includes
every neighbor tied at the kth distance, avoiding arbitrary index-based choices
on symmetric structures. Edges have dx, dy and distance. Six neighbors is a
development hyperparameter, independent of the earlier K=6 cluster count. This
choice needs later radius/k sensitivity analysis. No node cutoff silently drops
outer peaks, and no lattice type is assumed in connectivity.

**Spatial graph:** one node corresponds to one of the 380 original atomic sites.
Edges link nearby sample positions in HAADF coordinates. The local encoder's
embedding can later become this graph's node feature. Reference graphs have no
corresponding HAADF spatial layout, so these two graph levels are never mixed.
The current stage exports spatial connectivity but does not train a context or
interface head. Avoiding unconditional spatial smoothing keeps future interfaces
from being erased before their labels are defined.

## 3. Schema comes from the producer, not guessed image axes

The upstream `vpcffit.extract_peak_coords` returns relative
`(y, x, sigma_y, sigma_x, ellipticity)` records [2]. The first two fields already
subtract the vPCF origin. The adapter swaps them to internal `(x,y)` and never
subtracts the stored origin a second time. JSON `$array` records and flattened
H5 lists with `peaks_shapes` use this explicit contract. Unknown schemas,
nonfinite entries, duplicate coordinates and malformed shapes fail loudly.

The original H5 atomic-position convention remains separate: `(x,y)` in the
HAADF image, consistent with the repository's existing mapper. Those absolute
coordinates are metadata and spatial-graph inputs, not local-encoder features.
This prevents the local library matcher from learning an image-location shortcut.

## 4. Preserve orientation unless the task authorizes invariance

Library keys distinguish phase family and viewing direction. The default
`orientation_aware` encoder therefore receives local displacement components and
edge directions. It is permutation invariant at graph readout, but does not
promise rotation invariance. No rotations or reflections are imposed as
label-preserving training augmentations by default.

The optional `invariant` mode removes vector components from the learned
features and uses radial/distance information. It supports a controlled symmetry
ablation, with rigid-rotation and reflection tests. Distance-based invariants
and equivariant models are motivated by the geometric-learning literature [3],
but this encoder is **not an EGNN implementation**: it has no equivariant
coordinate-update layer. Reflection invariance can erase chirality, and distance
invariance can erase orientation information. Choosing it requires a scientific
decision about the target labels. Small floating-point tolerances still apply.

## 5. Shared encoding and class prototypes

The same local graph adapter and encoder process references and image samples.
Two message-passing layers use learned edge messages, degree-normalized
aggregation and residual node updates. Mean node pooling gives permutation-
invariant graph readout. Log node count preserves a small amount of information
about observed peak support. A 32-dimensional normalized embedding is the output.
Hidden and embedding dimensions are explicit configuration settings.

Peak count can also encode crop support or detector settings. It is a possible
shortcut, so a peak-count-only baseline and matched-support ablation belong in
validation before interpreting a class match as structural recognition.

Each reference class has a prototype, the mean embedding of its references,
following the metric-learning rationale of prototypical networks [4]. The current
file has one reference per phase/view class, so its clean prototype is that
reference's embedding. Squared embedding distance gives a nearest-reference
candidate. It is not a calibrated posterior, and no automatic unknown threshold
is fitted from training distances.

The fitting routine uses two independently jittered training views per original
reference and minimizes prototype cross-entropy. All 51 classes participate in
this development fit. The jitter standard deviation is 0.002 normalized units,
equivalent to 0.256 source pixels at the default scale. This is a diagnostic
augmentation magnitude, not a measured experimental noise model. Graph topology
is frozen during this augmentation, while displacement/radius/edge features are
recomputed. Independent simulation data are needed to validate augmentation.

## 6. What makes the geometry objective physically motivated

A decoder predicts each clean reference edge length from the noisy graph's node
representations. The auxiliary loss is mean squared error in normalized length:

`L = L_reference_prototype + 0.1 * L_denoising_edge_distance`.

The target is observed reference geometry. The decoder does not receive the
clean target length directly. This encourages preservation of measured geometric
relationships. It is a representation-learning constraint, not a governing
equation or proof of physical consistency. Geometry is also present in encoder
inputs, so this auxiliary task may be easy and requires an ablation to establish
any benefit beyond the graph representation.

Physics-informed modeling should tie constraints to actual physical knowledge
[5]. No energy/force residual is introduced: there is no supplied interatomic
potential, validated material model or complete 3D structure for one. No boundary
loss is trained because aligned interface labels are missing. Those omissions
are deliberate limits on the current scientific claims, not hidden fallback paths.

## 7. Units, strain and domain transfer

The current files lack a verified shared pixel-size metadata contract. The
upstream generator exposes a vPCF pixel-size parameter, but that default does
not prove the settings of these saved files. Dividing both sources by 128 is
fixed numeric conditioning, not physical unit conversion or per-graph rescaling.
It does not fit statistics to the 380 image graphs or normalize away every
sample's isotropic strain. Physical spacing, crop support, peak detection and
noise differences may still cause cross-domain mismatch.

The default 128 places the inspected roughly 100–500-pixel displacement values
at order-one magnitudes. It is an explicit conditioning choice rather than a
learned calibration; freeze or select it within a proper training/validation
protocol before any later generalization experiment.

Candidate exports therefore carry `physical_units_verified=false`. Confirm
source calibration and acquisition/generation settings before interpreting
nearest-reference labels physically. Never rename unvalidated matches to A/B/C
or AB/AC/BC merely because six groups were requested previously.

## 8. Reproducibility and evidence

Graphs use numeric NPZ arrays and JSON metadata with `allow_pickle=False`.
Checkpoints contain a state dictionary, prototypes, class order and configuration
and reload with `weights_only=True` [7]. Source hashes, raw-source hashes, sample
IDs and configuration travel with the run. The development fit records real
parameter movement and verifies a new model instance reproduces the saved
reference embeddings. Tests also cover message isolation between graph batches.

No validation/test accuracy is manufactured from augmented copies of a single
reference. The evaluation plan in [EVALUATION.md](EVALUATION.md) follows the
separation principles for structured data [6].

Numbered citations resolve to [REFERENCES.md](REFERENCES.md). All architectural
sizes, graph k, loss weights and augmentation settings above are project design
choices requiring later validation, rather than values established by those papers.
