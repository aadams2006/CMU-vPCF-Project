from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pignn.graphs import peak_graph,neighbor_edges,edge_features,save_graphs,load_graphs
from pignn.encoder import ReferenceEncoder,batch_graphs,prototypes,nearest_prototypes
from pignn.splits import group_split,spatial_bands,within_split_edges
from pignn.training import train_reference
from pignn.data import library_graphs,h5_graphs


def fixture(label='A',factor=1.):
    peaks=np.array([[1,2,.2,.3,1.5],[-2,1,.3,.3,1],[3,-1,.2,.3,1.5],[-1,-3,.2,.2,1.]])
    peaks[:,:2]*=factor
    return peak_graph(peaks,dict(sample_id=label,label=label,parent_group=label),coordinate_scale=1.,neighbors=2)


class GraphTests(unittest.TestCase):
    def test_source_adapters_preserve_labels_ids_and_axis_conventions(self):
        import h5py
        config=dict(coordinate_scale=1.,neighbors=2,peak_schema='vpcffit_y_x_sigma_y_sigma_x_ellipticity')
        peaks=[[1.,2.,.2,.3,1.5],[-2.,1.,.3,.3,1.]]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'library.json';path.write_text(json.dumps({'Phase_001':{'$array':peaks}}))
            refs=library_graphs(path,config)
            self.assertEqual(refs[0].metadata['label'],'Phase_001')
            self.assertEqual(refs[0].metadata['parent_group'],'Phase')
            path=Path(d)/'samples.h5'
            with h5py.File(path,'w') as f:
                g=f.create_group('experiments');g['atomic_positions']=[[10.,20.]]
                g['vpcf_origin']=[[500.,498.]];g['peaks_shapes']=[[2,5]]
                g['vpcf_peaks']=np.array(peaks).reshape(1,-1)
            samples=h5_graphs(path,config)
            self.assertEqual(samples[0].metadata['sample_idx'],0)
            self.assertEqual(samples[0].metadata['atomic_xy'],[10.,20.])
            np.testing.assert_array_equal(samples[0].pos[1],[2.,1.])
            np.testing.assert_array_equal(refs[0].pos,samples[0].pos)

    def test_upstream_yx_and_already_centered_contract(self):
        g=fixture();np.testing.assert_array_equal(g.pos[1],[2,1]);np.testing.assert_array_equal(g.pos[0],[0,0])
        self.assertEqual(g.x[0,3],1);self.assertTrue(np.all(g.x[1:,3]==0))

    def test_invalid_schema_nan_and_duplicates_rejected(self):
        with self.assertRaises(ValueError):peak_graph([[1,2,3,4,1]],{},schema='xy_guess')
        with self.assertRaises(ValueError):peak_graph([[np.nan,2,3,4,1]],{})
        with self.assertRaises(ValueError):neighbor_edges([[0,0],[0,0]])

    def test_symmetric_edges_no_loops_and_all_kth_ties(self):
        edges=neighbor_edges([[0,0],[1,0],[-1,0],[0,1],[0,-1]],1)
        pairs=set(map(tuple,edges.T));self.assertEqual(len([1 for a,b in pairs if a==0]),4)
        self.assertTrue(all(a!=b and (b,a) in pairs for a,b in pairs))

    def test_archive_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'g.npz';graphs=[fixture(),fixture('B',2.)];save_graphs(p,graphs)
            restored=load_graphs(p)
            for a,b in zip(graphs,restored):
                self.assertEqual(a.metadata,b.metadata)
                for key in ['pos','x','edge_index','edge_attr']:np.testing.assert_array_equal(getattr(a,key),getattr(b,key))


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_node_permutation_invariance(self):
        torch.manual_seed(3);model=ReferenceEncoder();g=fixture();p=np.array([3,0,4,1,2]);inverse=np.argsort(p)
        shuffled=replace(g,pos=g.pos[p],x=g.x[p],edge_index=inverse[g.edge_index])
        with torch.no_grad():a=model(batch_graphs([g]))[0];b=model(batch_graphs([shuffled]))[0]
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)

    def test_invariant_mode_rigid_rotation_and_reflection(self):
        model=ReferenceEncoder(encoder_mode='invariant');g=fixture()
        angle=.37
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        for matrix in [np.array([[0,-1],[1,0]]),rotation,np.diag([-1,1])]:
            pos=(g.pos@matrix.T).astype(np.float32)
            transformed=replace(g,pos=pos,edge_attr=edge_features(pos,g.edge_index))
            with torch.no_grad():a=model(batch_graphs([g]))[0];b=model(batch_graphs([transformed]))[0]
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)

    def test_batch_graphs_do_not_exchange_messages(self):
        model=ReferenceEncoder();g=fixture();other=fixture('B',2)
        with torch.no_grad():alone=model(batch_graphs([g]))[0];together=model(batch_graphs([g,other]))[0]
        torch.testing.assert_close(alone[0],together[0],atol=2e-6,rtol=2e-6)

    def test_absolute_atomic_location_is_not_an_encoder_feature(self):
        model=ReferenceEncoder();g=fixture();moved=replace(g,metadata=dict(g.metadata,atomic_xy=[999,999]))
        with torch.no_grad():a=model(batch_graphs([g]))[0];b=model(batch_graphs([moved]))[0]
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_training_and_checkpoint_reload(self):
        cfg=dict(seed=4,hidden_dim=8,embedding_dim=8,message_layers=1,encoder_mode='orientation_aware',
            epochs=3,temperature=.1,jitter_std=.002,geometry_weight=.1,learning_rate=.001)
        graphs=[fixture('A',1),fixture('B',2),fixture('C',3)]
        model,z,classes,centers,history,movement=train_reference(graphs,cfg)
        self.assertGreater(movement,0);self.assertEqual(len(history),3);self.assertTrue(np.isfinite([h['loss'] for h in history]).all())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'weights.pt';torch.save(model.state_dict(),path)
            fresh=ReferenceEncoder(**cfg);fresh.load_state_dict(torch.load(path,weights_only=True))
            with torch.no_grad():restored=fresh(batch_graphs(graphs))[0]
            torch.testing.assert_close(z,restored)
        best,_=nearest_prototypes(z,centers);self.assertEqual(best.shape,(3,))

    def test_prototypes_average_same_class_only(self):
        classes,centers=prototypes(torch.tensor([[0.,0.],[2.,2.],[9.,9.]]),['A','A','B'])
        self.assertEqual(classes,['A','B']);torch.testing.assert_close(centers,torch.tensor([[1.,1.],[9.,9.]]))


class SplitTests(unittest.TestCase):
    def test_parent_leakage_and_duplicate_ids_rejected(self):
        records=[dict(sample_id='a',parent_group='parent')]
        with self.assertRaises(ValueError):group_split(records,['parent'],['parent'],[])
        with self.assertRaises(ValueError):group_split(records*2,['parent'],[],[])

    def test_spatial_buffer_and_cross_split_edge_removal(self):
        pos=np.array([[0,0],[4,0],[5,0],[6,0],[10,0],[15,0],[16,0],[17,0],[20,0]])
        assignment=spatial_bands(pos,(5,16),1.)
        self.assertTrue(np.all(assignment[[1,2,3,5,6,7]]==-1))
        edges=neighbor_edges(pos,2);kept=within_split_edges(edges,assignment,0)
        self.assertTrue(np.all(assignment[kept]==0))
        with self.assertRaises(ValueError):spatial_bands(pos,(5,16),0)


if __name__=='__main__':unittest.main()
