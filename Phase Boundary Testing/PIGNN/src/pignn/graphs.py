"""Explicit vpcffit peak schema, graph construction, and non-pickle archives."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

SCHEMA='vpcffit_y_x_sigma_y_sigma_x_ellipticity'


@dataclass
class Graph:
    pos: np.ndarray
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    metadata: dict


def neighbor_edges(pos, neighbors=6):
    """Symmetric directed kNN, including all kth-distance ties, no self loops."""
    pos=np.asarray(pos,dtype=np.float64)
    if pos.ndim!=2 or pos.shape[1]!=2 or not np.isfinite(pos).all():
        raise ValueError('Expected finite Nx2 coordinates')
    if neighbors<1 or int(neighbors)!=neighbors:raise ValueError('neighbors must be a positive integer')
    if len(np.unique(pos,axis=0))!=len(pos):raise ValueError('Duplicate coordinates')
    if len(pos)<2:return np.empty((2,0),dtype=np.int64)
    distances=cdist(pos,pos);np.fill_diagonal(distances,np.inf)
    k=min(neighbors,len(pos)-1)
    threshold=np.partition(distances,k-1,axis=1)[:,k-1]
    adjacency=distances<=threshold[:,None]+1e-9
    adjacency|=adjacency.T;np.fill_diagonal(adjacency,False)
    return np.array(np.nonzero(adjacency),dtype=np.int64)


def edge_features(pos,edges):
    delta=pos[edges[1]]-pos[edges[0]]
    return np.column_stack([delta,np.linalg.norm(delta,axis=1)]).astype(np.float32)


def peak_graph(peaks,metadata,coordinate_scale=128.,neighbors=6,schema=SCHEMA):
    if schema!=SCHEMA:raise ValueError('Unknown peak schema; refuse to guess columns')
    a=np.asarray(peaks,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=5 or len(a)==0 or not np.isfinite(a).all():
        raise ValueError('Peaks must be a nonempty finite Nx5 array')
    if not np.isfinite(coordinate_scale) or coordinate_scale<=0:raise ValueError('Invalid coordinate scale')
    if (a[:,2:4]<0).any() or (a[:,4]<1-0.02).any():raise ValueError('Invalid sigma or ellipticity fields')
    # Upstream values already subtract the vPCF origin. Convert (y,x) to (x,y).
    keep=np.linalg.norm(a[:,:2],axis=1)>1e-8
    retained=a[keep]
    if not len(retained):raise ValueError('No noncentral peaks')
    pos=np.vstack([np.zeros((1,2)),retained[:,[1,0]]/coordinate_scale]).astype(np.float32)
    x=np.column_stack([np.linalg.norm(pos,axis=1),
        np.r_[0.,retained[:,2:4].mean(1)/coordinate_scale],
        np.r_[1.,retained[:,4]],np.r_[1.,np.zeros(len(retained))]]).astype(np.float32)
    edges=neighbor_edges(pos,neighbors)
    meta=dict(metadata,schema=schema,coordinate_scale=coordinate_scale,
              coordinate_units='source_pixels_uncalibrated',dropped_origin_peaks=int((~keep).sum()),
              raw_peak_count=len(a),node_features=['radius','mean_sigma','ellipticity','is_origin'])
    return Graph(pos,x,edges,edge_features(pos,edges),meta)


def perturb(graph,rng,std):
    """Noise-only diagnostic augmentation; topology frozen, synthetic origin fixed."""
    pos=graph.pos.copy();pos[1:]+=rng.normal(0,std,pos[1:].shape).astype(np.float32)
    x=graph.x.copy();x[:,0]=np.linalg.norm(pos,axis=1)
    return replace(graph,pos=pos,x=x,edge_attr=edge_features(pos,graph.edge_index))


def save_graphs(path,graphs):
    if not graphs:raise ValueError('Empty graph collection')
    ptr=np.r_[0,np.cumsum([len(g.pos) for g in graphs])]
    edge_ptr=np.r_[0,np.cumsum([g.edge_index.shape[1] for g in graphs])]
    np.savez_compressed(path,pos=np.concatenate([g.pos for g in graphs]),x=np.concatenate([g.x for g in graphs]),
        edge_index=np.concatenate([g.edge_index+ptr[i] for i,g in enumerate(graphs)],axis=1),
        edge_attr=np.concatenate([g.edge_attr for g in graphs]),graph_ptr=ptr,edge_ptr=edge_ptr,
        metadata=np.array([json.dumps(g.metadata,sort_keys=True) for g in graphs]))


def load_graphs(path):
    with np.load(path,allow_pickle=False) as a:
        result=[]
        for i,meta in enumerate(a['metadata']):
            start,stop=a['graph_ptr'][i:i+2];es,ee=a['edge_ptr'][i:i+2]
            result.append(Graph(a['pos'][start:stop].copy(),a['x'][start:stop].copy(),
                a['edge_index'][:,es:ee].copy()-start,a['edge_attr'][es:ee].copy(),json.loads(str(meta))))
    return result
