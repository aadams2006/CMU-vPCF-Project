"""Explicit group/spatial separation. No automatic split of augmented views."""
import numpy as np


def group_split(records,train_groups,val_groups,test_groups):
    sets=[set(train_groups),set(val_groups),set(test_groups)]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Parent groups overlap across splits')
    ids=[r['sample_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate sample identities')
    result={name:[] for name in ['train','validation','test']}
    for i,r in enumerate(records):
        hits=[j for j,s in enumerate(sets) if r['parent_group'] in s]
        if len(hits)!=1:raise ValueError('Each parent group must belong to exactly one split')
        result[list(result)[hits[0]]].append(i)
    return result


def spatial_bands(atomic_xy,cuts,footprint_radius):
    """Reserve R on either side of each cut, giving >=2R cross-split separation."""
    pos=np.asarray(atomic_xy,dtype=float);a,b=cuts
    if pos.ndim!=2 or pos.shape[1]!=2 or not np.isfinite(pos).all():raise ValueError('Invalid positions')
    if not 0<footprint_radius or not np.isfinite(footprint_radius) or not a<b:
        raise ValueError('Explicit positive physical sample-footprint radius and ordered cuts required')
    x=pos[:,0];assignment=np.full(len(pos),-1,dtype=int)
    assignment[x<a-footprint_radius]=0
    assignment[(x>a+footprint_radius)&(x<b-footprint_radius)]=1
    assignment[x>b+footprint_radius]=2
    if any(not np.any(assignment==i) for i in range(3)):raise ValueError('Empty spatial split after buffering')
    return assignment


def within_split_edges(edge_index,assignment,split):
    edges=np.asarray(edge_index)
    return edges[:,(assignment[edges[0]]==split)&(assignment[edges[1]]==split)]
