"""Auditable PyTorch DEC/IDEC; no training fallback. Labels never enter fit()."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class Config:
    latent_dim: int = 3
    n_clusters: int = 3
    clustering_weight: float = 1.0
    pretrain_epochs: int = 75
    learning_rate: float = 0.001
    cluster_epochs: int = 150
    update_interval: int = 5


class DeepClustering(nn.Module):
    def __init__(self, input_dim, config, model_name):
        super().__init__()
        if model_name not in ('DEC', 'IDEC'):
            raise ValueError('Only DEC and IDEC are supported; no fallback exists.')
        self.config, self.model_name = config, model_name
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(),
                                     nn.Linear(32, 16), nn.ReLU(),
                                     nn.Linear(16, config.latent_dim))
        self.decoder = nn.Sequential(nn.Linear(config.latent_dim, 16), nn.ReLU(),
                                     nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))
        self.centers = nn.Parameter(torch.empty(config.n_clusters, config.latent_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x):
        z = self.encoder(x)
        q = (1 + (z[:, None] - self.centers[None]).square().sum(2)).reciprocal()
        q = q / q.sum(1, keepdim=True)
        return z, q

    @staticmethod
    def target(q):
        w = q.square() / q.sum(0).clamp_min(1e-12)
        return (w / w.sum(1, keepdim=True)).detach()

    @torch.no_grad()
    def infer(self, x):
        self.eval()
        z, q = self(torch.as_tensor(x, dtype=torch.float32))
        return z.numpy(), q.numpy(), q.argmax(1).numpy()


def fit(x, config, model_name, seed, output_dir):
    """AE pretraining then Student-t KL refinement (plus reconstruction for IDEC)."""
    if min(config.pretrain_epochs, config.cluster_epochs, config.update_interval) < 1:
        raise ValueError('Training epochs and update interval must be positive.')
    if config.clustering_weight <= 0 or config.learning_rate <= 0:
        raise ValueError('Loss weight and learning rate must be positive.')
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = []
    def log(event, **fields):
        record = dict(event=event, model=model_name, seed=seed, **fields)
        events.append(record)
        with (out / 'runtime.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    log('code_path', backend='torch', implementation=__file__, fallback=False,
        path='autoencoder_pretrain -> kmeans_initialization_only -> gradient_refinement -> learned_encoder_evaluation',
        config=asdict(config))
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError('Expected finite two-dimensional features.')
    model = DeepClustering(x.shape[1], config, model_name)
    ae_parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(ae_parameters, lr=config.learning_rate)
    history = []
    for epoch in range(config.pretrain_epochs):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model.decoder(model.encoder(x)), x)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite pretraining loss')
        loss.backward()
        optimizer.step()
        history.append(dict(stage='pretrain', epoch=epoch, loss=loss.item()))
    log('pretrain_complete', steps=config.pretrain_epochs, reconstruction_loss=loss.item())
    with torch.no_grad():
        z = model.encoder(x).numpy()
    initializer = KMeans(n_clusters=config.n_clusters, n_init=20, random_state=seed).fit(z)
    with torch.no_grad():
        model.centers.copy_(torch.from_numpy(initializer.cluster_centers_))
    centers_before = model.centers.detach().clone()
    encoder_before = torch.cat([p.detach().flatten().clone() for p in model.encoder.parameters()])
    decoder_before = torch.cat([p.detach().flatten().clone() for p in model.decoder.parameters()])
    np.save(out / 'initial_centers.npy', centers_before.numpy())
    np.save(out / 'initial_assignments.npy', initializer.labels_)
    log('centers_initialized', method='KMeans', role='initialization_only')
    parameters = list(model.encoder.parameters()) + [model.centers]
    if model_name == 'IDEC':
        parameters += list(model.decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    gradient_max = 0.0
    for epoch in range(config.cluster_epochs):
        if epoch % config.update_interval == 0:
            with torch.no_grad():
                p = model.target(model(x)[1])
        optimizer.zero_grad()
        z, q = model(x)
        kl = nn.functional.kl_div(q.clamp_min(1e-12).log(), p, reduction='batchmean')
        reconstruction = nn.functional.mse_loss(model.decoder(z), x)
        loss = config.clustering_weight * kl
        if model_name == 'IDEC':
            loss = loss + reconstruction
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite clustering loss; no fallback')
        loss.backward()
        if model.centers.grad is None or not torch.isfinite(model.centers.grad).all():
            raise RuntimeError('Missing or invalid center gradient')
        gradient_max = max(gradient_max, model.centers.grad.norm().item())
        optimizer.step()
        history.append(dict(stage='clustering', epoch=epoch, loss=loss.item(),
                            kl=kl.item(), reconstruction=reconstruction.item()))
    center_delta = (model.centers.detach() - centers_before).norm().item()
    encoder_delta = (torch.cat([p.detach().flatten() for p in model.encoder.parameters()]) - encoder_before).norm().item()
    decoder_delta = (torch.cat([p.detach().flatten() for p in model.decoder.parameters()]) - decoder_before).norm().item()
    if min(center_delta, encoder_delta, gradient_max) <= 0:
        raise RuntimeError('No verified encoder/center training; refusing result')
    if model_name == 'IDEC' and decoder_delta <= 0:
        raise RuntimeError('IDEC decoder did not train')
    z, q, pred = model.infer(x.numpy())
    if not np.isfinite(z).all() or not np.isfinite(q).all():
        raise FloatingPointError('Invalid learned model outputs')
    proof = dict(cluster_steps=config.cluster_epochs, center_delta=center_delta,
                 encoder_delta=encoder_delta, decoder_delta=decoder_delta,
                 max_center_gradient=gradient_max, fallback=False,
                 prediction_source='Student_t_argmax', embedding_source='trained_encoder',
                 changed_assignments=int(np.sum(pred != initializer.labels_)))
    log('training_verified', **proof)
    torch.save(dict(state_dict=model.state_dict(), config=asdict(config),
                    model_name=model_name, input_dim=x.shape[1], seed=seed), out / 'model.pt')
    np.savez_compressed(out / 'learned_outputs.npz', embeddings=z, probabilities=q,
                        predictions=pred, centers=model.centers.detach().numpy())
    (out / 'loss_history.json').write_text(json.dumps(history), encoding='utf-8')
    return model, proof
