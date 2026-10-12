"""Rerun prior frozen configurations changing only K, then map both runs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from deep_benchmark import Config, fit
from metrics import aggregate_numeric_metrics, pairwise_stability
from map_synthetic_clusters import map_run
from run_deep_benchmark import score, save_csv


def run(baseline, output, k):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with np.load(baseline / 'final_benchmark.npz') as data:
        arrays = {key: data[key] for key in data.files}
    np.savez_compressed(output / 'final_benchmark.npz', **arrays)
    x, y, shape = arrays['features'], arrays['ground_truth'], tuple(arrays['shape'])
    configs = json.loads((baseline / 'selected_configs.json').read_text())
    for config in configs.values():
        config['n_clusters'] = k
    (output / 'selected_configs.json').write_text(json.dumps(configs, indent=2))
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x).astype('float32')
    np.savez(output / 'final_scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    rows, aggregates, stability = [], [], []
    for name in ['KMeans', 'DEC', 'IDEC']:
        runs, model_rows = {}, []
        for seed in [42, 43, 44, 45, 46]:
            print(f'FINAL {name} K={k} seed={seed}', flush=True)
            folder = output / 'final' / f'{name}_{seed}'
            if name == 'KMeans':
                folder.mkdir(parents=True)
                model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(x)
                pred, z, proof = model.labels_, x, {}
                np.savez_compressed(folder / 'learned_outputs.npz', predictions=pred,
                                    centers=model.cluster_centers_)
            else:
                model, proof = fit(xs, Config(**configs[name]), name, seed, folder)
                z, q, pred = model.infer(xs)
            row = dict(model=name, seed=seed, requested_k=k, **score(z, pred, y, shape, k), **proof)
            rows.append(row)
            model_rows.append(row)
            runs[seed] = pred
            save_csv(rows, output / 'final_metrics_by_seed.csv')
        aggregates.extend(dict(model=name, **a) for a in aggregate_numeric_metrics(model_rows))
        stability.extend(dict(model=name, **s) for s in pairwise_stability(runs))
    save_csv(aggregates, output / 'final_metrics_aggregate.csv')
    save_csv(stability, output / 'final_pairwise_stability.csv')
    manifest = dict(baseline=str(baseline), requested_k=k, final_seeds=[42,43,44,45,46],
                    protocol='Saved selected configurations; only cluster count changed; no new search',
                    dataset_sha256=hashlib.sha256((baseline/'final_benchmark.npz').read_bytes()).hexdigest(),
                    torch=torch.__version__, configs=configs)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    map_run(baseline, output / 'map_back_k3')
    map_run(output, output / f'map_back_k{k}')
    lines = ['# Fixed K comparison', '',
             'Same saved synthetic dataset and five seeds as the K=3 baseline. '
             'Only K changed; saved DEC/IDEC hyperparameters were frozen without retuning. '
             'Raw boundary metrics retain extra cluster edges. Ground truth contains three phases.', '',
             '| Model | Mean ARI | Mean raw boundary F1 | Occupied clusters per seed |',
             '|---|---:|---:|---|']
    for name in ['KMeans','DEC','IDEC']:
        group = [r for r in rows if r['model']==name]
        lines.append(f"| {name} | {np.mean([r['adjusted_rand_index'] for r in group]):.5f} | "
                     f"{np.mean([r['raw_boundary_f1'] for r in group]):.5f} | "
                     f"{[r['occupied_clusters'] for r in group]} |")
    lines.extend(['', 'Both map-back directories include all 15 final model/seed results, '
                  'coordinate CSVs, raw label arrays and PNGs. The synthetic feature-channel overlay '
                  'is not registered to the presentation STEM image. Six requested clusters '
                  'do not imply six occupied or physically identified environments.'])
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=ROOT/'Results/Deep_Clustering/run_20260908')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=6)
    args = parser.parse_args()
    if args.k < 2:
        parser.error('K must be at least two')
    run(args.baseline, args.output, args.k)
