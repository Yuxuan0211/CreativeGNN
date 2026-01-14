from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint

from .graph_ode import GraphNeuralODEFunc
from .local_solver import RegionLocalSolver
from .flux_layer import FluxLayer
from .mlp import build_mlp


class CreativeGNN(nn.Module):
    """Encode → continuous graph dynamics → region local solver → flux correction → decode."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_regions: int = 4,
        ode_steps: int = 4,
        ode_method: str = "rk4",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = build_mlp(in_dim + 3, hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        self.ode_func = GraphNeuralODEFunc(hidden_dim, hidden_dim, dropout=dropout)
        self.local_solver = RegionLocalSolver(hidden_dim, num_regions=num_regions, dropout=dropout)
        self.flux_layer = FluxLayer(hidden_dim, dropout=dropout)
        self.decoder = build_mlp(hidden_dim, hidden_dim, out_dim, num_layers=3, dropout=dropout)
        self.ode_steps = ode_steps
        self.ode_method = ode_method

        self.register_buffer("integration_times", torch.linspace(0, 1, steps=ode_steps + 1))

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_in = torch.cat([batch.x, batch.pos], dim=-1)
        h0 = self.encoder(x_in)

        self.ode_func.set_graph(batch.pos, batch.edge_index)
        times = self.integration_times.to(h0.device)
        h_traj = odeint(self.ode_func, h0, times, method=self.ode_method)
        h_latent = h_traj[-1]

        h_local = self.local_solver(h_latent, batch.edge_index, batch.region_mask)
        h_flux, flux_edges, flux_residual = self.flux_layer(h_local, batch.pos, batch.edge_index)

        pred = self.decoder(h_flux)
        aux = {
            "latent": h_flux,
            "flux_edges": flux_edges,
            "flux_residual": flux_residual,
        }
        return pred, aux
