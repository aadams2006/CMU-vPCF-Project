"""Independent saved-artifact checks for the full-380 original-image experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(run):
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['samples']==manifest['training_samples']==380
    assert manifest['test_samples']==0 and manifest['k']==6
    with np.load(run/'dataset.npz') as data:
        x=data['normalized_features'];features=data['features'];positions=data['coordinates']
        np.testing.assert_array_equal(data['training_sample_idx'],np.arange(380))
        low=features.min(0,keepdims=True);scale=features.max(0,keepdims=True)-low;scale[scale==0]=1
        np.testing.assert_array_equal(low,data['minimum'])
        np.testing.assert_array_equal(scale,data['range'])
        np.testing.assert_array_equal(x,(features-low)/scale)
    assert x.shape==(380,7) and positions.shape==(380,2)
    checked=[]
    for seed in manifest['seeds']:
        for model in ['dec','idec']:
            folder=run/f'seed_{seed}'/model
            proof=json.loads((folder/'training_verification.json').read_text())
            assert proof['training_samples']==380 and not proof['initialized_from_previous_weights']
            assert proof['reload_predictions_match']
            assert proof['gradient_updates']==manifest['refinement_steps']
            assert proof['autoencoder_epochs']==manifest['pretrain_epochs']
            assert min(proof[k] for k in ['pretraining_encoder_delta','refinement_encoder_delta','center_delta'])>0
            with np.load(folder/'learned_outputs.npz') as data:
                pred=data['predictions'];q=data['probabilities'];centers=data['centers']
                assert q.shape==(380,6) and centers.shape==(6,6)
                assert np.isfinite(q).all() and (q>=0).all()
                np.testing.assert_allclose(q.sum(1),1,atol=1e-5)
                np.testing.assert_array_equal(pred,q.argmax(1))
            frame=pd.read_csv(folder/f'{model}_cluster_labels.csv')
            np.testing.assert_array_equal(frame.sample_idx,np.arange(380))
            assert frame.was_training_sample.all()
            np.testing.assert_array_equal(frame.cluster,pred)
            mapped=pd.read_csv(folder/'mapped_coordinates.csv')
            np.testing.assert_allclose(mapped[['atomic_x','atomic_y']],positions)
            np.testing.assert_array_equal(mapped.cluster,pred)
            with np.load(folder/'cluster_regions.npz') as regions:
                grid=regions['cluster_grid'];nearest=regions['nearest_sample_idx']
                np.testing.assert_array_equal(grid,pred[nearest])
                np.testing.assert_array_equal(grid[positions[:,1].astype(int),positions[:,0].astype(int)],pred)
            history=pd.read_csv(folder/'refinement_history.csv')
            assert len(history)==manifest['refinement_steps']
            assert np.isfinite(history[['kl','reconstruction_mse','objective']]).all().all()
            weights=folder/f'{model.upper()}_model_final.weights.h5'
            assert weights.stat().st_size>1000000
            for name in ['cluster_regions.png','all_sample_cluster_labels.png','training_and_population.png']:
                assert (folder/name).stat().st_size>10000
            checked.append(dict(seed=seed,model=model,occupied_clusters=len(np.unique(pred)),
                weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest()))
    report=dict(passed=True,model_runs=len(checked),samples_per_run=380,held_out_test=False,checks=checked)
    (run/'artifact_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    verify(parser.parse_args().run)
