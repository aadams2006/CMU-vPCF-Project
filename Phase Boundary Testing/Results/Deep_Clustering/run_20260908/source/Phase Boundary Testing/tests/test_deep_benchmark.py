"""Runtime checks for actual optimization, inference provenance, and failure behavior."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from deep_benchmark import Config, DeepClustering, fit
from run_deep_benchmark import raw_boundary_metrics


class DeepBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng=np.random.default_rng(12)
        cls.x=np.vstack([rng.normal(i,0.2,(30,6)) for i in [-2,0,2]]).astype('float32')
        cls.config=Config(pretrain_epochs=4,cluster_epochs=8)

    def test_both_models_train_and_reload_exact_outputs(self):
        for name in ['DEC','IDEC']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                model,proof=fit(self.x,self.config,name,11,tmp)
                self.assertGreater(proof['center_delta'],0)
                self.assertGreater(proof['encoder_delta'],0)
                self.assertEqual(proof['cluster_steps'],8)
                self.assertEqual(proof['decoder_delta']>0,name=='IDEC')
                z,q,pred=model.infer(self.x)
                with np.load(Path(tmp)/'learned_outputs.npz') as saved:
                    np.testing.assert_array_equal(saved['embeddings'],z)
                np.testing.assert_array_equal(pred,q.argmax(1))
                checkpoint=torch.load(Path(tmp)/'model.pt',weights_only=True)
                restored=DeepClustering(6,self.config,name)
                restored.load_state_dict(checkpoint['state_dict'])
                np.testing.assert_array_equal(restored.infer(self.x)[0],z)
                events=[json.loads(line) for line in (Path(tmp)/'runtime.jsonl').read_text().splitlines()]
                self.assertEqual(events[-1]['event'],'training_verified')
                self.assertFalse(events[-1]['fallback'])

    def test_initialization_failure_propagates_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, patch('deep_benchmark.KMeans',side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError,'test failure'):
                fit(self.x,self.config,'DEC',11,tmp)
            self.assertFalse((Path(tmp)/'model.pt').exists())

    def test_target_formula_and_detachment(self):
        q=torch.tensor([[0.8,0.2],[0.3,0.7]],requires_grad=True)
        p=DeepClustering.target(q)
        w=q.detach().numpy()**2/q.detach().numpy().sum(0)
        np.testing.assert_allclose(p.numpy(),w/w.sum(1,keepdims=True))
        self.assertFalse(p.requires_grad)

    def test_extra_cluster_edges_are_penalized(self):
        truth=np.zeros((10,12),dtype=int);truth[:,6:]=1
        pred=truth.copy();pred[:5,:6]=2
        exact=raw_boundary_metrics(truth.ravel(),truth.ravel(),truth.shape)
        extra=raw_boundary_metrics(truth.ravel(),pred.ravel(),truth.shape)
        self.assertEqual(exact['raw_boundary_iou'],1)
        self.assertLess(extra['raw_boundary_iou'],1)
        self.assertLess(extra['raw_boundary_f1'],1)


if __name__=='__main__': unittest.main()
