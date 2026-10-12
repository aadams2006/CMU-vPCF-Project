"""Map H5 frame assignments onto the original DM3 using repository mapper."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from ncempy.io import dm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent/'Experimentally-obtained vPCF Testing/vPCF_test_2/src/vpcf_to_image_mapper.py'


def map_run(run, output):
    output.mkdir(parents=True, exist_ok=False)
    h5 = ROOT/'Data/SyntheticModel_HfO_80pm_vPCFs_65.h5'
    dm3 = ROOT/'Data/SyntheticModel_HfO_80pm_gaussian_HAADF.dm3'
    image = dm.dmReader(str(dm3))['data']
    with h5py.File(h5) as handle:
        group = handle['experiments']
        positions = group['atomic_positions'][:]
        origins = group['vpcf_origin'][:]
        peaks = group['peaks_shapes'][:]
        vshape = group['vpcf_images'].shape[1:]
    # Existing mapper defines atomic_positions[0]=x, [1]=y.
    if (positions < 0).any() or (positions[:,0] >= image.shape[1]).any() or (positions[:,1] >= image.shape[0]).any():
        raise ValueError('Atomic coordinates outside original DM3')
    metadata = dict(total_samples=len(positions), samples={str(i):dict(
        sample_id=i, atomic_positions=p.tolist(), vpcf_origin=origins[i].tolist(),
        vpcf_shape=list(vshape), peaks_shape=peaks[i].tolist(), crystal_label='unlabeled')
        for i,p in enumerate(positions)})
    metafile = output/'all_samples_metadata.json'
    metafile.write_text(json.dumps(metadata, indent=2))
    spec = importlib.util.spec_from_file_location('reference_mapper', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summaries = []
    files = sorted(run.rglob('*_cluster_labels.csv'))
    if not files:
        raise ValueError('No saved cluster labels')
    for label_file in files:
        method = label_file.name.split('_cluster_labels')[0]
        labels = pd.read_csv(label_file)
        ids = labels.sample_idx.to_numpy(dtype=int)
        if len(set(ids)) != len(ids) or (ids<0).any() or (ids>=len(positions)).any():
            raise ValueError('Invalid H5 frame indices')
        folder = output/label_file.parent.relative_to(run)
        folder.mkdir(parents=True, exist_ok=True)
        # Current trainer writes cluster_raw; legacy mapper expects cluster.
        if 'cluster' not in labels and 'cluster_raw' in labels:
            labels = labels.rename(columns={'cluster_raw': 'cluster'})
        labels[['sample_idx','cluster']].to_csv(folder/f'{method}_cluster_labels.csv',index=False)
        mapper = module.VPCFToImageMapper(str(metafile), str(folder), method)
        frame = mapper.export_cluster_coordinates(str(folder/'mapped_coordinates.csv'))
        np.testing.assert_allclose(frame[['atomic_x','atomic_y']], positions[frame.sample_id])
        expected = labels.set_index('sample_idx').loc[frame.sample_id,'cluster'].to_numpy()
        np.testing.assert_array_equal(frame.cluster, expected)
        np.savez_compressed(folder/'mapped_assignments.npz', sample_idx=frame.sample_id,
                            x=frame.atomic_x, y=frame.atomic_y, cluster=frame.cluster,
                            source_image_shape=image.shape)
        fig,ax = plt.subplots(figsize=(6,6))
        ax.imshow(image, cmap='gray', origin='upper')
        ax.scatter(positions[:,0],positions[:,1],s=12,facecolors='none',edgecolors='gray',alpha=.4)
        plotted = ax.scatter(frame.atomic_x,frame.atomic_y,c=frame.cluster,
                             cmap='tab10',vmin=0,vmax=9,s=40,edgecolors='white',linewidths=.5)
        fig.colorbar(plotted,ax=ax,ticks=sorted(frame.cluster.unique()),label='Raw cluster ID')
        ax.set(title=f'{method.upper()} · {len(frame)}/{len(positions)} H5 frames mapped',
               xlabel='Original image x (pixels)',ylabel='Original image y (pixels)')
        fig.tight_layout()
        fig.savefig(folder/'original_haadf_map_back.png',dpi=200)
        plt.close(fig)
        summaries.append(dict(model=method,path=str(label_file.relative_to(run)),
                              mapped_samples=len(frame),total_source_samples=len(positions),
                              occupied_clusters=int(frame.cluster.nunique())))
    pd.DataFrame(summaries).to_csv(output/'mapping_summary.csv',index=False)
    manifest = dict(source_h5=str(h5),source_dm3=str(dm3),coordinate_convention='x=atomic_positions[:,0], y=atomic_positions[:,1], image origin upper; existing mapper convention',
                    registration='Original DM3 pixel coordinates; no resizing or screenshot registration',
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [h5,dm3,REFERENCE]},
                    label_sources=str(run), mapped_runs=summaries)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'README.md').write_text('# Original synthetic H5 map-back\n\n'
        'Assignments join to original H5 frame indices and atomic_positions through the existing '
        'VPCFToImageMapper, then plot on the original DM3 without rescaling. '
        'Gray circles mark all source coordinates; colored circles mark tested frames. '
        'The original smoke test uses the first 12 of 380 frames, one pretraining epoch and '
        'two refinement iterations. This is a pipeline smoke test, not a full boundary study. '
        'Cluster IDs have no established bulk/interface semantics.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    map_run(args.run,args.output)
