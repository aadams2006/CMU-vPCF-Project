# Primary sources and their role

Accessed October 25, 2026. The cited methods motivate design choices; this package
does not claim to reproduce their published benchmarks.

1. **Gilmer et al. (2017), Neural Message Passing for Quantum Chemistry.**
   [ICML/PMLR paper](https://proceedings.mlr.press/v70/gilmer17a.html).
   Basis for learned edge messages, aggregation and graph-level readout.
   Our two-layer dimensions, degree normalization and feature choices are project
   choices, not a reproduction of that paper's chemical model.

2. **vpcffit producer source and project.**
   [Repository](https://github.com/vpcffit/vpcffit),
   [extract_peak_coords source](https://github.com/vpcffit/vpcffit/blob/4e9a4bdc5290914b9fb413c5a0deecf47b69f206/vpcffit/vpcffit.py),
   [serialization utilities](https://github.com/vpcffit/vpcffit/blob/4e9a4bdc5290914b9fb413c5a0deecf47b69f206/vpcffit/utils.py).
   Local checkout inspected at commit `4e9a4bdc5290914b9fb413c5a0deecf47b69f206`.
   Establishes origin-relative y/x peak fields, Gaussian width/ellipticity
   semantics, and `$array` JSON serialization. This is the format reference, not
   independent proof of the H5's generation settings or physical calibration.

3. **Satorras, Hoogeboom and Welling (2021), E(n) Equivariant Graph Neural Networks.**
   [ICML/PMLR paper](https://proceedings.mlr.press/v139/satorras21a.html).
   Motivates careful geometric symmetry handling. Our optional distance-only
   scalar encoder is invariant for tested transforms of its input graph; it is
   not the EGNN coordinate-update architecture. Default directional mode does
   not force rotation or reflection invariance.

4. **Snell, Swersky and Zemel (2017), Prototypical Networks for Few-shot Learning.**
   [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html).
   Basis for shared embeddings and class means as reference prototypes.
   Using one stored template per class does not establish few-shot generalization.

5. **Karniadakis et al. (2021), Physics-informed machine learning.**
   [Nature Reviews Physics](https://doi.org/10.1038/s42254-021-00314-5).
   Supports tying constraints to specified physical knowledge. The implemented
   edge-distance denoising loss is geometric representation regularization.
   Governing-equation, energy and force residuals are outside the current scope.

6. **Roberts et al. (2017), Cross-validation strategies for data with temporal,
   spatial, hierarchical, or phylogenetic structure.**
   [Original publication](https://doi.org/10.1111/ecog.02881).
   Basis for separating dependent parents/regions rather than random augmented
   samples. Buffer size still requires this dataset's actual sample footprint.

7. **PyTorch serialization documentation.**
   [Serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html),
   [torch.load](https://docs.pytorch.org/docs/stable/generated/torch.load.html).
   Basis for state-dictionary checkpoints and restricted loading. The actual
   runtime version is recorded with each training run.

## Local data evidence

- The selected simulation JSON contains 51 exact phase/view keys in six crystal
  families. Both maxIndex1 and maxIndex2 are present; only maxIndex1 is selected
  by default, avoiding accidental orientation-library mixing.
- The original H5 contains 380 atomic positions, fitted peak arrays and stored
  origins. An inspected first peak array has shape 68×5 and signed displacement
  columns. These are direct file observations, not facts inferred from papers.
- File hashes, graph counts and sample identities are recorded in the prepared
  run's `input_manifest.json` and `graph_inventory.json`.
