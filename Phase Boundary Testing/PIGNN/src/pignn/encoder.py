"""Native PyTorch message passing and reference prototypes; no PyG dependency."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def batch_graphs(graphs):
    ptr=np.r_[0,np.cumsum([len(g.pos) for g in graphs])]
    return dict(x=torch.tensor(np.concatenate([g.x for g in graphs]),dtype=torch.float32),
        pos=torch.tensor(np.concatenate([g.pos for g in graphs]),dtype=torch.float32),
        edge_index=torch.tensor(np.concatenate([g.edge_index+ptr[i] for i,g in enumerate(graphs)],axis=1),dtype=torch.long),
        edge_attr=torch.tensor(np.concatenate([g.edge_attr for g in graphs]),dtype=torch.float32),
        batch=torch.repeat_interleave(torch.arange(len(graphs)),torch.tensor(np.diff(ptr))),n_graphs=len(graphs))


class ReferenceEncoder(nn.Module):
    def __init__(self,hidden_dim=32,embedding_dim=32,message_layers=2,encoder_mode='orientation_aware',**unused):
        super().__init__()
        if encoder_mode not in ['orientation_aware','invariant']:raise ValueError('Unknown encoder mode')
        self.mode=encoder_mode
        self.input=nn.Sequential(nn.Linear(6 if self.mode=='orientation_aware' else 4,hidden_dim),nn.SiLU())
        edge_dim=3 if self.mode=='orientation_aware' else 1
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden_dim+edge_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim)) for _ in range(message_layers)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.SiLU(),nn.LayerNorm(hidden_dim)) for _ in range(message_layers)])
        self.readout=nn.Sequential(nn.Linear(hidden_dim+1,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,embedding_dim))
        self.geometry=nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1),nn.Softplus())

    def forward(self,data):
        x=data['x'];edge=data['edge_attr']
        if self.mode=='orientation_aware':x=torch.cat([x,data['pos']],1)
        else:edge=edge[:,2:3]
        h=self.input(x);src,dst=data['edge_index']
        degree=torch.bincount(dst,minlength=len(h)).clamp_min(1).to(h.dtype)[:,None]
        for message,update in zip(self.messages,self.updates):
            m=message(torch.cat([h[src],h[dst],edge],1))
            summed=torch.zeros_like(h).index_add(0,dst,m)/degree
            h=h+update(torch.cat([h,summed],1))
        pooled=h.new_zeros((data['n_graphs'],h.shape[1])).index_add(0,data['batch'],h)
        counts=torch.bincount(data['batch'],minlength=data['n_graphs']).to(h.dtype)[:,None]
        z=F.normalize(self.readout(torch.cat([pooled/counts,counts.log()/10],1)),dim=1)
        distances=self.geometry(torch.cat([h[src],h[dst]],1)).squeeze(1)
        return z,distances


def prototypes(embeddings,labels):
    classes=sorted(set(labels))
    if not classes:raise ValueError('No reference classes')
    return classes,torch.stack([embeddings[[i for i,y in enumerate(labels) if y==label]].mean(0) for label in classes])


def nearest_prototypes(embeddings,centers):
    distances=torch.cdist(embeddings,centers).square()
    best=distances.argmin(1)
    return best,distances
