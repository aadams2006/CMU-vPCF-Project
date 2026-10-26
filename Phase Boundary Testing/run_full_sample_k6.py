"""Fresh DEC/IDEC K=6 fitting on every original H5 sample, with audited maps.

This is transductive clustering of one complete image, not held-out evaluation.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from ncempy.io import dm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from dec import DEC
from idec import IDEC
from vpcf_data_loader import extract_statistical_features
from metrics import compute_internal_metrics,aggregate_numeric_metrics,pairwise_stability
from cluster_region_figures import nearest_sample_regions,save_region_figure,save_atomic_figure


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def weights_vector(variables):
    return np.concatenate([v.numpy().ravel() for v in variables])


def run(output,seeds,pretrain_epochs,refine_steps):
    output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    h5=ROOT/'Data/SyntheticModel_HfO_80pm_vPCFs_65.h5'
    dm3=ROOT/'Data/SyntheticModel_HfO_80pm_gaussian_HAADF.dm3'
    background=dm.dmReader(str(dm3))['data']
    features=[]
    with h5py.File(h5) as f:
        g=f['experiments'];images=g['vpcf_images']
        positions=g['atomic_positions'][:];origins=g['vpcf_origin'][:];peaks=g['peaks_shapes'][:]
        shape=images.shape[1:]
        if len(images)!=380 or positions.shape!=(380,2):raise ValueError('Expected all 380 original samples')
        for i in range(len(images)):
            image=np.nan_to_num(np.asarray(images[i],dtype=np.float32),nan=0.,posinf=0.,neginf=0.)
            features.append(extract_statistical_features(image[None])[0])
    features=np.asarray(features,dtype=np.float32)
    low=features.min(0,keepdims=True);scale=features.max(0,keepdims=True)-low;scale[scale==0]=1
    x=(features-low)/scale
    if not np.isfinite(x).all():raise ValueError('Nonfinite inputs')
    np.savez_compressed(output/'dataset.npz',features=features,normalized_features=x,coordinates=positions,
                        training_sample_idx=np.arange(380),minimum=low,range=scale)
    pd.DataFrame(features,columns=['mean','std','min','max','median','skewness','excess_kurtosis']).to_csv(output/'features.csv',index_label='sample_idx')
    metadata={'total_samples':380,'samples':{str(i):dict(sample_id=i,atomic_positions=p.tolist(),
        vpcf_origin=origins[i].tolist(),vpcf_shape=list(shape),peaks_shape=peaks[i].tolist(),
        crystal_label='unlabeled') for i,p in enumerate(positions)}}
    write_json(output/'all_samples_metadata.json',metadata)
    mapper_path=ROOT.parent/'Experimentally-obtained vPCF Testing/vPCF_test_2/src/vpcf_to_image_mapper.py'
    spec=importlib.util.spec_from_file_location('reference_mapper',mapper_path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    source_files=[Path(__file__),ROOT/'src/dec.py',ROOT/'src/idec.py',ROOT/'src/vpcf_data_loader.py',
                  ROOT/'src/metrics.py',ROOT/'src/cluster_region_figures.py',mapper_path]
    (output/'source').mkdir()
    for p in source_files:shutil.copy2(p,output/'source'/p.name)
    manifest=dict(protocol='Fresh transductive full-image clustering; no held-out split or generalization claim',
        samples=380,training_samples=380,test_samples=0,ground_truth_available=False,k=6,seeds=seeds,
        architecture=[7,500,500,2000,6],feature_method='statistical',normalization='minmax fitted on all 380',
        pretrain_epochs=pretrain_epochs,pretrain_batch_size=64,pretrain_optimizer='Adam default lr=0.001',
        refinement_steps=refine_steps,refinement_batch_size=380,target_update_interval=5,
        refinement_optimizer='SGD lr=0.01',idec_gamma=0.1,early_stopping=False,
        initialization='fresh random network then KMeans n_init=20 on pretrained embeddings',
        python=platform.python_version(),tensorflow=tf.__version__,numpy=np.__version__,
        source_data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [h5,dm3]},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(output/'manifest.json',manifest)
    rows=[];all_labels={};aggregates=[];stability=[]
    for method,cls in [('dec',DEC),('idec',IDEC)]:
        label_runs={};model_rows=[]
        for seed in seeds:
            print(f'TRAIN {method.upper()} seed={seed}, all 380 samples',flush=True)
            tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
            folder=output/f'seed_{seed}'/method;folder.mkdir(parents=True)
            model=cls(dims=manifest['architecture'],n_clusters=6,save_dir=str(folder),random_state=seed)
            model.autoencoder.compile(optimizer='adam',loss='mse')
            initialized=weights_vector(model.encoder.trainable_variables)
            ae_history=model.autoencoder.fit(x,x,epochs=pretrain_epochs,batch_size=64,shuffle=True,verbose=0)
            write_json(folder/'pretrain_history.json',ae_history.history)
            pretrained=weights_vector(model.encoder.trainable_variables)
            z=model.encoder(x,training=False).numpy()
            km=KMeans(n_clusters=6,n_init=20,random_state=seed).fit(z)
            model.model.get_layer('clustering').set_weights([km.cluster_centers_])
            initial_centers=km.cluster_centers_.copy()
            np.savez_compressed(folder/'initialization.npz',embeddings=z,predictions=km.labels_,centers=initial_centers)
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=.01))
            history=[]
            for step in range(refine_steps):
                if step%5==0:
                    raw=model.model(x,training=False)
                    q=(raw[0] if method=='idec' else raw).numpy()
                    target=model.target_distribution(q).astype(np.float32)
                losses=model.model.train_on_batch(x,[target,x] if method=='idec' else target)
                raw=model.model(x,training=False)
                q=(raw[0] if method=='idec' else raw).numpy()
                kl=float(np.mean(np.sum(target*np.log(np.maximum(target,1e-12)/np.maximum(q,1e-12)),axis=1)))
                mse=float(np.mean((model.autoencoder(x,training=False).numpy()-x)**2))
                objective=kl if method=='dec' else .1*kl+mse
                if not np.isfinite([kl,mse,objective]).all():raise FloatingPointError('Nonfinite loss')
                history.append(dict(step=step+1,kl=kl,reconstruction_mse=mse,objective=objective))
            pd.DataFrame(history).to_csv(folder/'refinement_history.csv',index=False)
            pred=q.argmax(1);z=model.encoder(x,training=False).numpy()
            centers=model.get_cluster_centers()
            proof=dict(pretraining_encoder_delta=float(np.linalg.norm(pretrained-initialized)),
                refinement_encoder_delta=float(np.linalg.norm(weights_vector(model.encoder.trainable_variables)-pretrained)),
                center_delta=float(np.linalg.norm(centers-initial_centers)),
                gradient_updates=int(model.model.optimizer.iterations.numpy()),
                autoencoder_epochs=len(ae_history.history['loss']),training_samples=380,
                initialized_from_previous_weights=False)
            if min(proof['pretraining_encoder_delta'],proof['refinement_encoder_delta'],proof['center_delta'])<=0:
                raise RuntimeError('Training movement not verified')
            if proof['gradient_updates']!=refine_steps:raise RuntimeError('Incorrect update count')
            weights=folder/f'{method.upper()}_model_final.weights.h5';model.model.save_weights(str(weights))
            np.savez_compressed(folder/'learned_outputs.npz',predictions=pred,probabilities=q,embeddings=z,centers=centers)
            # Reload into a newly constructed architecture, not the in-memory fitted instance.
            reloaded=cls(dims=manifest['architecture'],n_clusters=6,save_dir=str(folder),random_state=seed)
            reloaded.load_weights(str(weights))
            raw=reloaded.model(x,training=False);qr=(raw[0] if method=='idec' else raw).numpy()
            np.testing.assert_allclose(q,qr,rtol=1e-6,atol=1e-7)
            np.testing.assert_array_equal(pred,qr.argmax(1))
            proof['reload_predictions_match']=True
            write_json(folder/'training_verification.json',proof)
            pd.DataFrame(dict(sample_idx=np.arange(380),cluster=pred,was_training_sample=True)).to_csv(folder/f'{method}_cluster_labels.csv',index=False)
            mapper=mod.VPCFToImageMapper(str(output/'all_samples_metadata.json'),str(folder),method)
            mapped=mapper.export_cluster_coordinates(str(folder/'mapped_coordinates.csv'))
            np.testing.assert_allclose(mapped[['atomic_x','atomic_y']],positions)
            np.testing.assert_array_equal(mapped.cluster,pred)
            grid,index,distance=nearest_sample_regions(positions,pred,background.shape)
            np.testing.assert_array_equal(grid[positions[:,1].astype(int),positions[:,0].astype(int)],pred)
            np.savez_compressed(folder/'cluster_regions.npz',cluster_grid=grid,nearest_sample_idx=index,nearest_sample_distance_px=distance)
            title=f'{method.upper()} · K = 6 · seed {seed} · trained on all 380 vPCFs'
            save_region_figure(background,grid,6,folder/'cluster_regions.png',title,positions,pred,
                'All 380 samples used in training · nearest-sample regions · no held-out test')
            save_atomic_figure(background,positions,pred,6,folder/'all_sample_cluster_labels.png',title)
            counts=np.bincount(pred,minlength=6)
            pd.DataFrame(dict(cluster=np.arange(6),samples=counts,fraction=counts/380)).to_csv(folder/'cluster_counts.csv',index=False)
            fig,axes=plt.subplots(1,3,figsize=(13,3.5))
            axes[0].plot(np.arange(1,pretrain_epochs+1),ae_history.history['loss']);axes[0].set(title='Autoencoder pretraining',xlabel='Epoch',ylabel='MSE')
            axes[1].plot(np.arange(1,refine_steps+1),[h['objective'] for h in history]);axes[1].set(title='Cluster refinement',xlabel='Update',ylabel='Objective (targets refreshed every 5)')
            axes[2].bar(np.arange(6),counts);axes[2].set(title='Cluster population',xlabel='Raw cluster ID',ylabel='Samples')
            fig.tight_layout();fig.savefig(folder/'training_and_population.png',dpi=170);plt.close(fig)
            metrics={f'embedding_{k}':v for k,v in compute_internal_metrics(z,pred).items()}
            metrics.update({f'input_{k}':v for k,v in compute_internal_metrics(x,pred).items()})
            row=dict(model=method,seed=seed,occupied_clusters=int((counts>0).sum()),
                min_cluster_samples=int(counts.min()),max_cluster_samples=int(counts.max()),**metrics,**proof)
            old=ROOT/f'Results/Full_Runs/synthetic_cluster_regions_20261012/k6/{method}/full_image_inference.npz'
            if old.exists():
                with np.load(old) as prior:row['agreement_with_12_sample_model_ari']=adjusted_rand_score(prior['predictions'],pred)
            write_json(folder/'metrics.json',row)
            rows.append(row);model_rows.append(row);label_runs[seed]=pred;all_labels[f'{method}_{seed}']=pred
            pd.DataFrame(rows).to_csv(output/'metrics_by_seed.csv',index=False)
            print(f'DONE {method.upper()} seed={seed}: cluster counts {counts.tolist()}',flush=True)
        aggregates.extend(dict(model=method,**a) for a in aggregate_numeric_metrics(model_rows))
        stability.extend(dict(model=method,**s) for s in pairwise_stability(label_runs))
    pd.DataFrame(aggregates).to_csv(output/'metrics_aggregate.csv',index=False)
    pd.DataFrame(stability).to_csv(output/'pairwise_seed_stability.csv',index=False)
    agreement=[]
    for seed in seeds:agreement.append(dict(seed=seed,DEC_IDEC_ARI=adjusted_rand_score(all_labels[f'dec_{seed}'],all_labels[f'idec_{seed}'])))
    pd.DataFrame(agreement).to_csv(output/'cross_model_agreement.csv',index=False)
    lines=['# Full-sample original-vPCF clustering, K=6','',
        'Fresh DEC and IDEC weights trained on all 380 original H5 vPCF samples. All assignments are from models fitted on those 380 samples. No 12-sample weights were reused.','',
        'This is transductive full-image clustering. There is no held-out split, so internal scores and seed stability do not measure generalization. No aligned physical ground-truth mask is available, so phase accuracy, interface F1 and boundary accuracy are not reported.','',
        '## Protocol','',f'Five seeds {seeds}. Seven statistical features per vPCF, min-max scaling fitted on all 380. Architecture 7–500–500–2000–6, mirrored autoencoder. '
        f'{pretrain_epochs} autoencoder epochs with Adam (batch 64), then {refine_steps} full-batch SGD updates (learning rate 0.01), targets refreshed every five updates. DEC minimizes KL; IDEC minimizes reconstruction MSE + 0.1 KL. No early stopping or hyperparameter selection on these results.','',
        'The earlier smoke test used 12 samples, one pretraining epoch and two refinement updates. Training coverage, duration and fitted scaling all changed here. Differences from that run cannot be attributed solely to adding samples. Statistical features summarize intensity distributions and do not retain full vPCF peak geometry.','',
        '## Final occupied clusters','', '| Model | Seed | Cluster counts, IDs 0–5 |','|---|---:|---|']
    for row in rows:
        counts=pd.read_csv(output/f'seed_{row["seed"]}'/row['model']/'cluster_counts.csv').samples.tolist()
        lines.append(f'| {row["model"].upper()} | {row["seed"]} | {counts} |')
    lines+=['','All seeds are saved without selection. Seed 42 is the fixed reference for viewing figures. Cluster IDs are arbitrary across runs. Six requested centers do not guarantee six occupied or physically distinct environments.','',
        '## Artifacts','',
        'Each seed/model directory contains final weights, initial and learned embeddings/centers, probabilities and assignments, both training histories, metrics, population counts, training/reload verification, original-image atomic and region maps, and a loss/population plot. The existing VPCFToImageMapper joins original sample IDs and atomic coordinates. Nearest-sample filling produces display regions without smoothing or semantic relabeling.','',
        'Region edges extend nearest atomic assignments over image pixels, including the image borders. They are visualization boundaries, not measured interfaces. No cluster text markers are drawn inside the image; legends identify all six cluster IDs.','',
        '## Reproduce','',
        '`python run_full_sample_k6.py --output Results/Full_Runs/<new-directory>` from Phase Boundary Testing, with repository dependencies plus TensorFlow and ncempy. Source H5/DM3 files remain local and are identified by SHA-256 in the manifest. Source scripts are snapshotted in source/.','',
        f'Elapsed run time: {time.time()-started:.1f} seconds. Every final model passed fresh-instance checkpoint reload and sample-to-region label round-trip checks.']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (output/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    inventory={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in output.rglob('*') if p.is_file() and p.name!='artifact_inventory.json'}
    write_json(output/'artifact_inventory.json',inventory)
    print('COMPLETE: fresh training, checkpoint reloads, metrics and image maps saved.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    parser.add_argument('--pretrain-epochs',type=int,default=50)
    parser.add_argument('--refine-steps',type=int,default=200)
    args=parser.parse_args()
    if min(args.pretrain_epochs,args.refine_steps)<1:parser.error('Epochs and updates must be positive')
    run(args.output,args.seeds,args.pretrain_epochs,args.refine_steps)
