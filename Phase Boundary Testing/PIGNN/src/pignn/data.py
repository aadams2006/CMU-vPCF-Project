"""Read reference and experimental peak lists without loading the image stack."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from .graphs import peak_graph,neighbor_edges,edge_features


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def library_graphs(path,config):
    data=json.loads(Path(path).read_text())
    if not isinstance(data,dict) or not data:raise ValueError('Expected a labeled reference dictionary')
    result=[]
    for label,record in sorted(data.items()):
        if not isinstance(record,dict) or set(record)!={'$array'}:
            raise ValueError('Expected the upstream $array serialization')
        family,view=label.rsplit('_',1)
        meta=dict(sample_id='library:'+label,label=label,crystal_family=family,view=view,
            parent_group=family,parent_group_basis='conservative_crystal_family_proxy',source=str(Path(path).resolve()))
        result.append(peak_graph(record['$array'],meta,config['coordinate_scale'],config['neighbors'],config['peak_schema']))
    return result


def h5_graphs(path,config):
    result=[]
    with h5py.File(path) as f:
        g=f['experiments'];n=len(g['atomic_positions'])
        for i in range(n):
            shape=tuple(int(v) for v in g['peaks_shapes'][i])
            if len(shape)!=2 or shape[1]!=5:raise ValueError('H5 peak schema mismatch')
            peaks=g['vpcf_peaks'][i].reshape(shape)
            meta=dict(sample_id=f'h5:{i:06d}',sample_idx=i,label=None,parent_group='synthetic_image_1',
                atomic_xy=g['atomic_positions'][i].astype(float).tolist(),
                vpcf_origin_yx=g['vpcf_origin'][i].astype(float).tolist(),source=str(Path(path).resolve()))
            result.append(peak_graph(peaks,meta,config['coordinate_scale'],config['neighbors'],config['peak_schema']))
    return result


def save_spatial_graph(path,graphs,neighbors=6):
    pos=np.asarray([g.metadata['atomic_xy'] for g in graphs],dtype=np.float64)
    idx=neighbor_edges(pos,neighbors)
    np.savez_compressed(path,atomic_xy=pos,edge_index=idx,edge_attr=edge_features(pos,idx),
        sample_idx=np.asarray([g.metadata['sample_idx'] for g in graphs]),
        sample_ids=np.asarray([g.metadata['sample_id'] for g in graphs]),units='original_HAADF_pixels')
    return idx
