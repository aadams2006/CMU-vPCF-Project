"""Map saved benchmark predictions using the existing sample-to-image mapper.

Coordinates refer to the generated benchmark grid, not the reference STEM image.
Raw cluster IDs are retained; ground truth never assigns semantic cluster names.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cluster_region_figures import save_region_figure

ROOT = Path(__file__).resolve().parents[1]
MAPPER = ROOT.parent / 'Experimentally-obtained vPCF Testing/vPCF_test_2/src/vpcf_to_image_mapper.py'


def map_run(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('reference_mapper', MAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(run / 'final_benchmark.npz') as data:
        coords = data['coordinates'].astype(int)
        shape = tuple(data['shape'].astype(int))
        truth = data['ground_truth']
        features = data['features']
    if len(np.unique(coords, axis=0)) != np.prod(shape) or len(truth) != len(coords):
        raise ValueError('Coordinates must cover the grid exactly once')
    if (coords < 0).any() or (coords >= np.array(shape)).any():
        raise ValueError('Coordinates outside benchmark grid')
    metadata = {'total_samples': len(coords), 'samples': {
        str(i): dict(sample_id=i, atomic_positions=[int(c), int(r)],
                     vpcf_origin=[int(c), int(r)], vpcf_shape=[1, 1],
                     peaks_shape='synthetic feature vector', crystal_label=int(truth[i]))
        for i, (r, c) in enumerate(coords)}}
    meta_file = output / 'all_samples_metadata.json'
    meta_file.write_text(json.dumps(metadata, indent=2))
    # Saved feature intensity is a benchmark channel, not a measured VPCF image.
    background = np.empty(shape)
    background[coords[:, 0], coords[:, 1]] = features[:, 0]
    np.save(output / 'synthetic_feature_channel_0.npy', background)
    records = []
    for folder in sorted((run / 'final').iterdir()):
        name, seed = folder.name.rsplit('_', 1)
        with np.load(folder / 'learned_outputs.npz') as data:
            labels = data['predictions'].astype(int)
            k = len(data['centers'])
        if len(labels) != len(coords) or (labels < 0).any() or (labels >= k).any():
            raise ValueError('Invalid sample assignments')
        dest = output / folder.name
        dest.mkdir()
        pd.DataFrame({'sample_idx': np.arange(len(labels)), 'cluster': labels}).to_csv(
            dest / f'{name.lower()}_cluster_labels.csv', index=False)
        mapper = module.VPCFToImageMapper(str(meta_file), str(dest), name.lower())
        frame = mapper.export_cluster_coordinates(str(dest / 'mapped_coordinates.csv'))
        grid = np.full(shape, -1, dtype=int)
        grid[frame.atomic_y.astype(int), frame.atomic_x.astype(int)] = frame.cluster
        np.testing.assert_array_equal(grid[coords[:, 0], coords[:, 1]], labels)
        np.save(dest / 'raw_cluster_map.npy', grid)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        axes[0].imshow(truth.reshape(shape), cmap='tab10', vmin=0, vmax=9)
        axes[0].set_title('Original synthetic phase grid')
        image = axes[1].imshow(grid, cmap='tab10', vmin=0, vmax=9)
        axes[1].set_title(f'{name}, K={k}, seed {seed}: raw IDs')
        fig.colorbar(image, ax=axes[1], ticks=range(k), label='Cluster ID')
        axes[2].imshow(background, cmap='gray')
        axes[2].imshow(grid, cmap='tab10', vmin=0, vmax=9, alpha=.45)
        axes[2].set_title('Overlay on synthetic feature channel 0')
        for ax in axes:
            ax.set(xlabel='Column (x)', ylabel='Row (y)')
        fig.tight_layout()
        fig.savefig(dest / 'map_back.png', dpi=180)
        plt.close(fig)
        save_region_figure(background,grid,k,dest/'cluster_regions.png',
                           f'{name} · K={k} · seed {seed} · synthetic spatial benchmark')
        records.append(dict(model=name, seed=int(seed), requested_k=k,
                            occupied_clusters=len(np.unique(labels)), mapped_samples=len(frame)))
    pd.DataFrame(records).to_csv(output / 'mapping_summary.csv', index=False)
    (output / 'README.md').write_text(
        '# Synthetic coordinate map-back\n\nAll saved final seeds are mapped by sample ID through '
        'the existing VPCFToImageMapper. x=column, y=row. Raw cluster IDs are retained. '
        'The overlay background is synthetic feature channel 0; it is not a measured STEM '
        'or VPCF image. No registration to the presentation image is available. '
        'Three phase ground-truth labels do not establish six bulk/interface identities.\n')
    return records
