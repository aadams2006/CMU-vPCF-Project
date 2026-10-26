# Input, graph and output contracts

## Read-only inputs

1. `../../Unused Data/data/saved_vpcfs/HfO_vPCFs_from_simulation_maxIndex1.json`
   relative to PIGNN. A dictionary of 51 exact phase/view labels, each containing
   one `{"$array": [[five numeric fields], ...]}` record.
2. `../Data/SyntheticModel_HfO_80pm_vPCFs_65.h5`. The 380-sample original H5.
   Required group: `/experiments`. Required datasets: `vpcf_peaks`,
   `peaks_shapes`, `atomic_positions`, `vpcf_origin`. The image stack is not read
   during graph preparation.

Source paths and SHA-256 values are recorded. No simulation-library source or
synced ChatGPT project file is modified. The maxIndex2 library is a separate,
larger orientation set and is not silently merged with the selected 51 entries.

## Peak columns

The one accepted schema is `vpcffit_y_x_sigma_y_sigma_x_ellipticity`:

| Column | Meaning | Handling |
|---|---|---|
| 0 | Origin-relative y displacement | Internal y after fixed-scale division |
| 1 | Origin-relative x displacement | Internal x after fixed-scale division |
| 2 | Fitted sigma_y | Average with sigma_x, divide by fixed scale |
| 3 | Fitted sigma_x | Average with sigma_y, divide by fixed scale |
| 4 | Ellipticity | Scalar shape descriptor, not peak intensity/confidence |

These meanings are established by the upstream producer source [2]. The H5
does not contain its own schema attribute, so application of that contract to
the H5 is an explicit, documented producer-format assumption. The inspected
numeric ranges are compatible with it; they do not independently prove units.
Confirm the producing notebook before quantitative physical interpretation.

An exactly central source peak is omitted and replaced by one marked synthetic
origin node. Near-central nonzero peaks are retained. Duplicate coordinates are
rejected, not silently averaged. Arrays must be finite, nonempty Nx5 with
nonnegative fitted widths and a plausible positive ellipticity. The small
ellipticity tolerance accommodates source rounding and does not certify the fit.

## Graph archive

`library_graphs.npz` and `sample_graphs.npz` contain:

- `pos`: concatenated Nx2 internal x/y displacement coordinates.
- `x`: Nx4 radius, mean sigma, ellipticity, origin-indicator features.
- `edge_index`: 2xE integer source/target indices, offset into the archive.
- `edge_attr`: Ex3 normalized dx, dy and Euclidean distance.
- `graph_ptr`, `edge_ptr`: graph/node/edge boundaries for lossless recovery.
- `metadata`: Unicode JSON records, not pickle objects.

Each graph has a stable sample ID. Library metadata retains the full label,
family/view strings and a conservative family-level parent proxy. H5 metadata
retains sample index, atomic x/y, the stored vPCF origin and the input source.
Atomic positions do not enter local encoder features.

`spatial_graph.npz` uses **HAADF pixels**, rather than vPCF displacement pixels.
Its `sample_idx` and `sample_ids` preserve the local-graph order exactly. Each
site can later consume the corresponding local embedding. The archive explicitly
labels its units. A spatial edge is a context relation, not a physical bond.

## Encoder artifacts

`reference_encoder.pt` includes the learned state dictionary, fixed configuration,
class order, prototypes and reference training IDs. `reference_embeddings.npz`
contains clean reference embeddings and prototypes. Training histories and a
manifest record what was fitted. No independent-validation fields are inferred.

`sample_embeddings.npz` contains all 380 embeddings, a squared-distance matrix
to the 51 reference prototypes, candidate indices, class order and atomic x/y.
`candidate_matches.json` joins candidates directly to original sample indices
and carries the status `unvalidated_library_candidate` for every sample. These
are not trained six-environment labels or calibrated unknown/known decisions.

## Expected errors

Unknown schema, incompatible checkpoint graph configuration, duplicate spatial
or local coordinates, malformed flattened peak arrays and nonfinite data raise
errors. No row is silently discarded to force a successful run. A failed run
must retain its failure evidence and use a new output directory on retry.

The source citations and the units distinction are explained in
[DESIGN.md](DESIGN.md) and [REFERENCES.md](REFERENCES.md).
