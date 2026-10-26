# Evaluation and next development steps

## Current checks

Software tests establish schema handling, symmetric connectivity, serialization,
permutation invariance, rigid symmetry in the optional invariant mode, isolated
graph batches, finite gradient fitting and fresh-instance checkpoint reload.
Actual-data preparation checks all 51 references and 380 local sample graphs.
These checks establish implementation behavior, not material-phase accuracy.

The reference development fit uses all 51 stored templates and noisy views of
those same templates. There are no independent held-out simulation examples.
Training loss, checkpoint replay and nearest-self-reference retrieval cannot be
used as claims of unseen classification performance.

The 380 image graphs are not training examples in this reference-fitting stage.
Their later candidate labels nevertheless remain unvalidated: acquisition/domain
calibration is unconfirmed and no aligned machine-readable physical label mask
has been supplied.

## Leakage controls already implemented

`group_split` rejects overlapping parent groups and duplicate sample identities.
Augmented versions of a template must remain with that template. Current family
groups are conservative proxies, not verified independent simulation identifiers.
Splitting them holds entire families out, which is an unknown-family experiment,
not a known-class test with every class represented in training.

`spatial_bands` requires an explicit positive sample-footprint radius R and
buffers each x boundary by R on each side. This yields at least 2R separation
between retained sample centers across splits. It rejects empty buffered splits.
`within_split_edges` removes edges crossing a split or touching the excluded
buffer. Future spatial context training must apply this filtering before any
message passing in an inductive evaluation.

The H5 does not specify the physical support used to calculate each local vPCF.
Consequently no default buffered split or generalization score is produced.
Estimate R from the producing workflow, not the vPCF raster's width: displacement
support and source-image crop support are different quantities. If vPCFs use
global information, a within-image spatial split may remain invalid. Independent
parent images/simulations are preferable. Structured-data validation motivates
this caution [6].

## Next experiments

1. Verify the H5 producer's peak schema, source calibration, displacement support
   and original-image neighborhood footprint. Align reference/image units and
   peak selection before interpreting candidates.
2. Obtain multiple independent parent simulations per intended phase/view label,
   including independent noise, distortions and imaging conditions. Retain
   parent IDs so all augmented/derived examples stay in the same split.
3. Define whether in-plane rotation, reflection, polarity and viewing direction
   preserve each target label. Compare direction-aware and invariant encoders
   under that policy, with the same parent-level split.
4. Establish aligned bulk/interface labels and widths. Library bulk labels alone
   cannot supervise AB/AC/BC interfaces. Save sample-ID labels rather than
   deriving truth from DEC/IDEC clusters or plot colors.
5. Fit the local encoder on training references only. Select dimensions, neighbor
   count/radius, noise magnitude and geometry-loss weight on validation parents.
   Evaluate untouched test parents once. Include a zero-geometry-loss ablation
   the existing statistical-feature baseline, a peak-count-only baseline, and
   matched reference/image peak-support settings.
6. Add a spatial context/interface head with buffered inductive splits, masking
   all cross-split edges and fitting preprocessing only on the training partition.
7. Calibrate known/unknown thresholds on held-out known and genuinely withheld
   structural classes. A nearest prototype always exists and is not evidence
   that a sample belongs to the library.

Energy or force penalties require a valid potential/material model, chemistry
and suitable structural information. They are not approximated by invented
functions in this foundation. References [3–6] motivate candidate methods and
evaluation principles, but do not validate this particular dataset or model.
