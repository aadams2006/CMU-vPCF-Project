"""Independent model searches followed by separate frozen-config seed runs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import itertools
import numpy as np
import pandas as pd
import torch
import sklearn
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from deep_benchmark import Config, fit
from metrics import (evaluate_clustering, boundary_map, aggregate_numeric_metrics,
                     pairwise_stability, optimal_label_mapping, remap_labels)
from Phase_Boundary_Training_Local_Fallback import generate_spatial_phase_benchmark


def raw_boundary_metrics(truth, pred, shape):
    """Label-invariant edges: never merge extra predicted clusters using truth."""
    a, b = boundary_map(truth.reshape(shape)), boundary_map(pred.reshape(shape))
    if not a.any() or not b.any():
        return dict(raw_boundary_f1=0., raw_boundary_iou=0., raw_boundary_distance_px=None)
    db, da = distance_transform_edt(~a)[b], distance_transform_edt(~b)[a]
    precision, recall = np.mean(db <= 1), np.mean(da <= 1)
    return dict(raw_boundary_f1=float(2*precision*recall/(precision+recall)) if precision+recall else 0.,
                raw_boundary_iou=float((a & b).sum()/(a | b).sum()),
                raw_boundary_distance_px=float(np.concatenate([db, da]).mean()))


def score(z, pred, y, shape, k, full=True):
    result = evaluate_clustering(z, pred, y_true=y, spatial_shape=shape) if full else dict(
        adjusted_rand_index=adjusted_rand_score(y, pred),
        normalized_mutual_info=normalized_mutual_info_score(y, pred))
    result.update(raw_boundary_metrics(y, pred, shape))
    counts = np.bincount(pred, minlength=k)
    result.update(cluster_counts=counts.tolist(), occupied_clusters=int((counts>0).sum()),
                  min_cluster_fraction=float(counts.min()/len(y)),
                  max_cluster_fraction=float(counts.max()/len(y)),
                  degenerate=bool((counts==0).any() or counts.min()/len(y)<0.01))
    return result


def save_csv(rows, path):
    pd.DataFrame([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows]).to_csv(path,index=False)


def save_data(out, name, data):
    x,y,shape,coords=data
    np.savez_compressed(out / f'{name}.npz', features=x, ground_truth=y, shape=shape, coordinates=coords)
    pd.DataFrame(dict(sample_idx=np.arange(len(y)), row=coords[:,0], column=coords[:,1],ground_truth=y)).to_csv(out/f'{name}_labels.csv',index=False)


def candidates(seed, trials):
    space = dict(latent_dim=[1,2,3,4], n_clusters=[2,3,4,5],
                 clustering_weight=[0.01,0.1,1.,10.], pretrain_epochs=[25,75,150],
                 learning_rate=[0.0003,0.001,0.003])
    all_configs = [dict(zip(space,v)) for v in itertools.product(*space.values())]
    rng=np.random.default_rng(seed)
    selected=[asdict(Config())]
    for i in rng.permutation(len(all_configs)):
        c=asdict(Config(**all_configs[i]))
        if c not in selected:
            selected.append(c)
        if len(selected)==trials: break
    return space,selected


def run(out, trials):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    data={name:generate_spatial_phase_benchmark(random_state=s) for name,s in [('search_train',2718),('search_validation',1618),('final_benchmark',314)]}
    for name,d in data.items(): save_data(out,name,d)
    # Verify byte-for-byte numerical equivalence to the committed earlier baseline.
    baseline=ROOT/'Results/Local_Fallback_Runs/run_kmeans_benchmark_20260827_194059/spatial_phase_boundary_test'
    old_x=pd.read_csv(baseline/'features.csv').drop(columns='sample_idx').to_numpy()
    old_y=pd.read_csv(baseline/'ground_truth_labels.csv').ground_truth.to_numpy()
    np.testing.assert_allclose(data['final_benchmark'][0],old_x,rtol=1e-14,atol=1e-14)
    np.testing.assert_array_equal(data['final_benchmark'][1],old_y)
    manifest=dict(python=platform.python_version(),torch=torch.__version__,sklearn=sklearn.__version__,
                  objective='maximize validation ARI; exclude empty or <1% clusters; ties prefer lower latent dimension',
                  search_training_seed=7,final_seeds=[42,43,44,45,46],trials_per_model=trials,
                  baseline_data_verified=True,device='cpu',threads=2,
                  dataset_sha256={n:hashlib.sha256((out/f'{n}.npz').read_bytes()).hexdigest() for n in data})
    try:
        manifest['base_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest['base_commit']='unavailable (source archive)'
    manifest['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__),ROOT/'src/deep_benchmark.py',ROOT/'src/metrics.py',ROOT/'Phase_Boundary_Training_Local_Fallback.py']}
    scaler=StandardScaler().fit(data['search_train'][0])
    train_x=scaler.transform(data['search_train'][0]).astype('float32')
    val_x=scaler.transform(data['search_validation'][0]).astype('float32')
    np.savez(out/'search_scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
    search_rows=[]; winners={}
    for model_name,search_seed in [('DEC',101),('IDEC',202)]:
        space,configs=candidates(search_seed,trials)
        manifest[model_name+'_search_space']=space
        model_rows=[]
        for i,c in enumerate(configs):
            ident=f'{model_name}_{i:02d}'
            print(f'SEARCH {ident} {c}',flush=True)
            row=dict(model=model_name,configuration=ident,seed=7,**c)
            try:
                model,proof=fit(train_x,Config(**c),model_name,7,out/'search'/ident)
                z,q,pred=model.infer(val_x)
                np.savez_compressed(out/'search'/ident/'validation_outputs.npz',embeddings=z,probabilities=q,predictions=pred)
                row.update(score(z,pred,data['search_validation'][1],data['search_validation'][2],c['n_clusters'],False))
                row.update(status='ok',**proof)
            except Exception as error:
                row.update(status='failed',error=repr(error),degenerate=True)
            search_rows.append(row);model_rows.append(row)
            save_csv(search_rows,out/'all_search_configurations.csv')
            print(f"RESULT {ident} ARI={row.get('adjusted_rand_index')} status={row['status']} counts={row.get('cluster_counts')}",flush=True)
        valid=[r for r in model_rows if r['status']=='ok' and not r['degenerate']]
        if not valid: raise RuntimeError(f'No nondegenerate {model_name} candidate')
        winner=sorted(valid,key=lambda r:(-r['adjusted_rand_index'],r['latent_dim'],r['configuration']))[0]
        winners[model_name]={k:winner[k] for k in asdict(Config())}
        manifest[model_name+'_winner']=winner
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'selected_configs.json').write_text(json.dumps(winners,indent=2),encoding='utf-8')
    # Final clustering is transductive, like the earlier K-means baseline. No labels enter training.
    x,y,shape,coords=data['final_benchmark']
    scaler=StandardScaler().fit(x); xs=scaler.transform(x).astype('float32')
    np.savez(out/'final_scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
    rows=[];aggregates=[];stability=[]
    for name in ['KMeans','DEC','IDEC']:
        label_runs={}; model_rows=[]
        for seed in manifest['final_seeds']:
            print(f'FINAL {name} seed={seed}',flush=True)
            if name=='KMeans':
                km=KMeans(n_clusters=3,n_init=20,random_state=seed).fit(x)
                pred=km.labels_;z=x;k=3;proof={}
                folder=out/'final'/f'{name}_{seed}';folder.mkdir(parents=True)
                np.savez_compressed(folder/'learned_outputs.npz',predictions=pred,centers=km.cluster_centers_)
            else:
                c=Config(**winners[name]);k=c.n_clusters
                model,proof=fit(xs,c,name,seed,out/'final'/f'{name}_{seed}')
                z,q,pred=model.infer(xs)
            r=dict(model=name,seed=seed,**score(z,pred,y,shape,k),**proof)
            rows.append(r);model_rows.append(r);label_runs[seed]=pred
            save_csv(rows,out/'final_metrics_by_seed.csv')
        aggregates.extend(dict(model=name,**a,variance=a['std']**2) for a in aggregate_numeric_metrics(model_rows))
        stability.extend(dict(model=name,**s) for s in pairwise_stability(label_runs))
    save_csv(aggregates,out/'final_metrics_aggregate.csv')
    save_csv(stability,out/'final_pairwise_stability.csv')
    old=pd.read_csv(baseline/'metrics_by_seed.csv')
    for r in rows:
        if r['model']=='KMeans':
            for metric in ['adjusted_rand_index','normalized_mutual_info','boundary_f1','boundary_iou_exact']:
                np.testing.assert_allclose(r[metric],old.loc[old.seed==r['seed'],metric].iloc[0],atol=1e-12)
    build_report(out,manifest,search_rows,rows,aggregates)
    make_figures(out)


def make_figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(out/'final_benchmark.npz') as d:
        y=d['ground_truth'];shape=tuple(d['shape'])
    fig,axes=plt.subplots(1,4,figsize=(13,3.6))
    axes[0].imshow(y.reshape(shape),vmin=0,vmax=2,cmap='viridis')
    axes[0].set_title('Saved ground truth')
    for ax,name in zip(axes[1:],['KMeans','DEC','IDEC']):
        with np.load(out/'final'/f'{name}_42'/'learned_outputs.npz') as d: pred=d['predictions']
        matched=remap_labels(pred,optimal_label_mapping(y,pred))
        ax.imshow(matched.reshape(shape),vmin=0,vmax=2,cmap='viridis')
        ax.set_title(f'{name} · seed 42')
    for ax in axes: ax.set_axis_off()
    fig.suptitle('Synthetic phase boundary · fixed reference seed (not selected for score)')
    fig.tight_layout();fig.savefig(out/'phase_maps.png',dpi=180);plt.close(fig)
    frame=pd.read_csv(out/'all_search_configurations.csv')
    fig,ax=plt.subplots(figsize=(10,4))
    for name,color in [('DEC','#3659a2'),('IDEC','#a34e1d')]:
        f=frame[frame.model==name]
        ax.plot(np.arange(len(f)),f.adjusted_rand_index,'o-',label=name,color=color,alpha=.8)
    ax.set(xlabel='Independent search configuration index',ylabel='Validation ARI',ylim=(0,1.02))
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'search_scores.png',dpi=180);plt.close(fig)


def build_report(out,manifest,search,rows,aggregates):
    metrics=['adjusted_rand_index','normalized_mutual_info','raw_boundary_f1','raw_boundary_iou','raw_boundary_distance_px']
    def summary(model,metric):
        r=next(a for a in aggregates if a['model']==model and a['metric']==metric)
        return f"{r['mean']:.5f} ± {r['std']:.5f}"
    lines=['# DEC / IDEC phase-boundary experiment','',
           'Actual CPU PyTorch training on the saved synthetic spatial phase benchmark (48×64, 3,072 samples, six features). Research H5/DM3 data were not used. The test features and labels match the committed earlier K-means benchmark; its metrics were reproduced to numerical tolerance.','',
           '## Protocol','',
           f'Separate searches: {manifest["trials_per_model"]} configurations per model, including one default configuration; the others are independently sampled with seeds 101 (DEC) and 202 (IDEC). Each candidate uses training seed 7. Selection maximizes ARI on an unseen noise realization (validation data seed 1618), after fitting on data seed 2718. Labels are saved and never passed into gradient training. Empty clusters or a cluster below 1% of validation samples disqualify a candidate. Exact ties prefer smaller latent dimension. This is a bounded random search, not proof of a global optimum. The geometries and phase means are shared across maps, so validation tests noise generalization, not new boundary geometries.','',
           'Final selected configurations are frozen before five fresh runs (42–46) on the original benchmark (data seed 314). Final fitting is transductive, matching the original K-means protocol. Final labels are used only for reporting, not selection. Uncertainty is across training seeds on a fixed dataset, not across physical specimens. The aggregate file includes sample variance and two-sided Student-t 95% confidence intervals for the mean (not clipped to metric bounds).','',
           'Architecture: 6→32→16→latent and mirrored decoder; ReLU hidden layers, linear latent/output. Standardization is fitted on training observations. Full-batch Adam; 150 refinement epochs; sharpened frequency-normalized targets refreshed every five epochs; Student-t alpha=1. DEC minimizes weight×KL(P||Q); IDEC minimizes MSE+weight×KL(P||Q). K-means initializes centers only, followed by gradient fitting. For DEC, scaling its sole loss is largely redundant under Adam, so the weight is searched as requested but is not a reconstruction tradeoff.','',
           'Internal scores use full learned encoder embeddings, without PCA. K-means uses original features to reproduce the baseline; silhouette values across these different spaces are not an apples-to-apples ranking. Final assignments come directly from Student-t argmax. Every run saves its checkpoint, latent embeddings, probabilities, fitted centers, initial centers, loss history, and runtime code-path evidence.','',
           '## Final comparison (mean ± sample SD, n=5)','',
           '| Model | ARI ↑ | NMI ↑ | Boundary F1 ↑ | Boundary IoU ↑ | Boundary distance px ↓ |',
           '|---|---:|---:|---:|---:|---:|']
    for name in ['KMeans','DEC','IDEC']:
        lines.append('| '+name+' | '+' | '.join(summary(name,m) for m in metrics)+' |')
    lines+=['','Boundaries use two-sided four-neighbor edges. F1 uses a one-pixel Euclidean tolerance; IoU is exact; distance is the mean of both directed boundary-distance samples. Raw scores preserve every predicted cluster edge. Existing phase-matched boundary scores are also retained for baseline compatibility; these can hide oversegmentation when k exceeds the number of phases.','', '## Selection and degeneracy','']
    best=max(['DEC','IDEC'],key=lambda n:manifest[n+'_winner']['adjusted_rand_index'])
    lines.append(f'**Preferred deep model by the predeclared validation objective: {best}.** This judgment uses validation ARI, not the final test scores. Compare final robustness and boundary scores below before adopting it.')
    for name in ['DEC','IDEC']:
        w=manifest[name+'_winner']
        lines += ['',f"{name}: {w['configuration']}, validation ARI {w['adjusted_rand_index']:.6f}; config `{json.dumps({k:w[k] for k in asdict(Config())})}`.",
                  f"Validation cluster counts: {w['cluster_counts']}; minimum fraction {w['min_cluster_fraction']:.4f}."]
        for r in rows:
            if r['model']==name:
                lines.append(f"- Seed {r['seed']}: occupied {r['occupied_clusters']}, sizes {r['cluster_counts']}, degenerate={r['degenerate']}; center movement {r['center_delta']:.6g}, encoder movement {r['encoder_delta']:.6g}.")
    lines+=['','## Every search configuration','', '| Model / ID | Latent | k | KL weight | Pretrain epochs | LR | Validation ARI | Raw boundary F1 | Counts | Status / degenerate |', '|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for r in search:
        lines.append(f"| {r['configuration']} | {r['latent_dim']} | {r['n_clusters']} | {r['clustering_weight']} | {r['pretrain_epochs']} | {r['learning_rate']} | {r.get('adjusted_rand_index',float('nan')):.6f} | {r.get('raw_boundary_f1',float('nan')):.6f} | {r.get('cluster_counts')} | {r['status']} / {r['degenerate']} |")
    lines+=['','## Reproduce','', '`python run_deep_benchmark.py --output <new-directory> --trials 24` from Phase Boundary Testing. Install requirements-deep.txt. Existing output directories are rejected to prevent mixing experiments.', '',
            'Definitions follow [DEC](https://proceedings.mlr.press/v48/xieb16.html) and [IDEC](https://www.ijcai.org/Proceedings/2017/243). This is a compact PyTorch implementation of their objectives, not a reproduction of their published large-network experiments.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--trials',type=int,default=24)
    args=p.parse_args()
    if args.trials<2: p.error('At least two search trials required')
    run(args.output,args.trials)
