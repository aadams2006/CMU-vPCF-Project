"""Augmented-reference fitting only; no held-out physical accuracy claim."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from .graphs import perturb
from .encoder import ReferenceEncoder,batch_graphs,prototypes


def train_reference(graphs,config):
    if config['epochs']<1 or config['temperature']<=0 or config['jitter_std']<0 or config['geometry_weight']<0:
        raise ValueError('Invalid training configuration')
    torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(config['seed'])
    model=ReferenceEncoder(**config)
    initial=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    labels=[g.metadata['label'] for g in graphs]
    if any(y is None for y in labels):raise ValueError('Unlabeled graph in reference fitting')
    classes=sorted(set(labels));targets=torch.tensor([classes.index(y) for y in labels])
    clean=batch_graphs(graphs);history=[]
    for epoch in range(config['epochs']):
        model.train();optimizer.zero_grad()
        support=batch_graphs([perturb(g,rng,config['jitter_std']) for g in graphs])
        query=batch_graphs([perturb(g,rng,config['jitter_std']) for g in graphs])
        zs,ds=model(support);zq,dq=model(query)
        _,centers=prototypes(zs,labels)
        classification=F.cross_entropy(-torch.cdist(zq,centers).square()/config['temperature'],targets)
        geometry=.5*(F.mse_loss(ds,clean['edge_attr'][:,2])+F.mse_loss(dq,clean['edge_attr'][:,2]))
        loss=classification+config['geometry_weight']*geometry
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reference loss')
        loss.backward();optimizer.step()
        history.append(dict(epoch=epoch+1,loss=float(loss.detach()),reference_class_loss=float(classification.detach()),
                            denoising_edge_distance_mse=float(geometry.detach())))
    movement=float(sum((v-initial[k]).square().sum() for k,v in model.state_dict().items()).sqrt())
    if movement<=0:raise RuntimeError('Encoder did not update')
    model.eval()
    with torch.no_grad():embeddings,_=model(clean)
    classes,centers=prototypes(embeddings,labels)
    return model,embeddings,classes,centers,history,movement
