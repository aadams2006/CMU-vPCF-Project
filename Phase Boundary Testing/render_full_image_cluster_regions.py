"""Infer all original H5 samples with saved models and render cluster regions.

Models are unchanged. Feature normalization is reconstructed from their original
training subset, never refitted to the full image. Region pixels use the nearest
sample label, so pixel boundaries are a visualization, not measured ground truth.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
from ncempy.io import dm

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from vpcf_data_loader import extract_statistical_features
from dec import DEC
from idec import IDEC
from cluster_region_figures import nearest_sample_regions,save_region_figure,save_atomic_figure


def render(output):
    output.mkdir(parents=True,exist_ok=False)
    h5=ROOT/'Data/SyntheticModel_HfO_80pm_vPCFs_65.h5'
    dm3=ROOT/'Data/SyntheticModel_HfO_80pm_gaussian_HAADF.dm3'
    image=dm.dmReader(str(dm3))['data']
    with h5py.File(h5) as handle:
        group=handle['experiments']
        positions=group['atomic_positions'][:]
        origins=group['vpcf_origin'][:]
        peaks=group['peaks_shapes'][:]
        images=group['vpcf_images']
        shape=images.shape[1:]
        features=[]
        for i in range(len(images)):
            frame=np.nan_to_num(np.asarray(images[i],dtype=np.float32),nan=0.,posinf=0.,neginf=0.)
            features.append(extract_statistical_features(frame[None])[0])
        features=np.asarray(features,dtype=np.float32)
    np.save(output/'statistical_features.npy',features)
    metadata={'total_samples':len(features),'samples':{str(i):dict(sample_id=i,
        atomic_positions=p.tolist(),vpcf_origin=origins[i].tolist(),vpcf_shape=list(shape),
        peaks_shape=peaks[i].tolist(),crystal_label='unlabeled') for i,p in enumerate(positions)}}
    meta_file=output/'all_samples_metadata.json'
    meta_file.write_text(json.dumps(metadata,indent=2))
    mapper_file=ROOT.parent/'Experimentally-obtained vPCF Testing/vPCF_test_2/src/vpcf_to_image_mapper.py'
    spec=importlib.util.spec_from_file_location('existing_mapper',mapper_file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=[]
    for k,source in [(3,ROOT/'Results/smoke_test'),(6,ROOT/'Results/Full_Runs/synthetic_h5_k6_20261012/seed_42')]:
        for method,cls in [('dec',DEC),('idec',IDEC)]:
            folder=source/method
            meta=json.loads((folder/f'{method}_metadata.json').read_text())
            config=meta['training_config'];n=meta['n_samples']
            assert config['feature_method']=='statistical' and config['normalize']=='minmax'
            assert config['n_clusters']==k
            low=features[:n].min(0,keepdims=True);high=features[:n].max(0,keepdims=True)
            scale=high-low;scale[scale==0]=1
            x=(features-low)/scale
            dest=output/f'k{k}'/method;dest.mkdir(parents=True)
            np.savez(dest/'training_normalization.npz',minimum=low,range=scale,training_samples=n)
            model=cls(dims=[features.shape[1],*config['hidden_dims'],k],n_clusters=k,save_dir=str(dest))
            weights=folder/f'{method.upper()}_model_final.weights.h5'
            model.load_weights(str(weights))
            raw=model.model.predict(x,verbose=0)
            q=raw[0] if method=='idec' else raw
            pred=q.argmax(1)
            saved=pd.read_csv(folder/f'{method}_cluster_labels.csv')
            label_col='cluster' if 'cluster' in saved else 'cluster_raw'
            original=saved[label_col].to_numpy(dtype=int)
            ids=saved.sample_idx.to_numpy(dtype=int)
            disagreements=int(np.sum(pred[ids]!=original))
            # Keep final-weight inference internally consistent, report any stale fit labels.
            pd.DataFrame(dict(sample_idx=np.arange(len(pred)),cluster=pred,
                was_training_sample=np.arange(len(pred))<n)).to_csv(dest/f'{method}_cluster_labels.csv',index=False)
            np.savez_compressed(dest/'full_image_inference.npz',probabilities=q,predictions=pred,
                coordinates=positions,was_training_sample=np.arange(len(pred))<n)
            mapper=module.VPCFToImageMapper(str(meta_file),str(dest),method)
            mapped=mapper.export_cluster_coordinates(str(dest/'mapped_coordinates.csv'))
            np.testing.assert_array_equal(mapped.cluster,pred)
            np.testing.assert_allclose(mapped[['atomic_x','atomic_y']],positions)
            grid,index,distance=nearest_sample_regions(positions,pred,image.shape)
            np.savez_compressed(dest/'cluster_regions.npz',cluster_grid=grid,
                                nearest_sample_idx=index,nearest_sample_distance_px=distance)
            np.testing.assert_array_equal(grid[positions[:,1].astype(int),positions[:,0].astype(int)],pred)
            title=f'{method.upper()} · K = {k} · all {len(pred)} original samples'
            save_region_figure(image,grid,k,dest/'cluster_regions.png',title,positions,pred,
                f'Nearest-sample regions · saved model trained on {n} samples; remaining {len(pred)-n} are inference')
            save_atomic_figure(image,positions,pred,k,dest/'all_sample_cluster_labels.png',title)
            counts=np.bincount(pred,minlength=k)
            record=dict(model=method,k=k,training_samples=n,inferred_samples=len(pred)-n,
                occupied_clusters=int((counts>0).sum()),cluster_counts=counts.tolist(),
                saved_training_label_disagreements=disagreements,
                weights=str(weights),weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
            records.append(record)
            (dest/'inference_manifest.json').write_text(json.dumps(record,indent=2))
            print(record,flush=True)
    pd.DataFrame([{**r,'cluster_counts':json.dumps(r['cluster_counts'])} for r in records]).to_csv(output/'summary.csv',index=False)
    (output/'README.md').write_text('# Original-image cluster regions\n\n'
        'K=3 and K=6 DEC/IDEC saved models are applied to all 380 original H5 samples. '
        'Models were trained on 12 samples; remaining 368 assignments are inference. '
        'No retraining, clustering on pixel coordinates, smoothing or semantic relabeling occurs. '
        'Min-max normalization uses the original 12 training frames. '
        'cluster_regions.png colors nearest-sample regions without text inside the image. '
        'All clusters appear in the legend. '
        'all_sample_cluster_labels.png colors every atomic sample like the reference. '
        'The inferred boundaries are display interpolation, not physical ground truth. '
        'IDs are model/run-specific; the same ID across runs need not mean the same phase. '
        'summary.csv records occupied counts and any disagreement between saved fit labels '
        'and reloaded final-weight inference.\n')
    inventory={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in output.rglob('*') if p.is_file()}
    (output/'artifact_inventory.json').write_text(json.dumps(inventory,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.output)
