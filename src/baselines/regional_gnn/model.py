from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.mlp import build_mlp


class RegionGatedLayer(nn.Module):
    """Region-masked message passing layer."""

    def __init__(self, hidden_dim: int, num_regions: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_regions = num_regions
        self.msg_mlp = build_mlp(hidden_dim * 2 + 3, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.region_mlps = nn.ModuleList(
            [build_mlp(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout) for _ in range(num_regions)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        rel_pos = pos[dst] - pos[src]
        msg = self.msg_mlp(torch.cat([h[src], h[dst], rel_pos], dim=-1))

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=h.device, dtype=h.dtype))
        agg = agg / (deg.unsqueeze(-1) + 1e-6)

        if region_mask is None:
            region_mask = torch.full((h.size(0), self.num_regions), 1.0 / self.num_regions, device=h.device, dtype=h.dtype)
        if region_mask.shape[1] != self.num_regions:
            r = region_mask.shape[1]
            if r < self.num_regions:
                pad = torch.zeros(h.size(0), self.num_regions - r, device=h.device, dtype=h.dtype)
                region_mask = torch.cat([region_mask, pad], dim=1)
            else:
                region_mask = region_mask[:, : self.num_regions]

        node_in = torch.cat([h, agg], dim=-1)
        update = torch.zeros_like(h)
        for ridx, mlp in enumerate(self.region_mlps):
            weight = region_mask[:, ridx].unsqueeze(-1)
            update = update + mlp(node_in) * weight
        return self.norm(h + update)


class RegionalGNN(nn.Module):
    """Regionalized GNN baseline with region-gated message passing."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_regions: int = 4,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = build_mlp(in_dim + 3, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.layers = nn.ModuleList(
            [RegionGatedLayer(hidden_dim, num_regions=num_regions, dropout=dropout) for _ in range(num_layers)]
        )
        self.decoder = build_mlp(hidden_dim, hidden_dim, out_dim, num_layers=2, dropout=dropout)

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_in = torch.cat([batch.x, batch.pos], dim=-1)
        h = self.encoder(x_in)
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.pos, batch.region_mask)
        pred = self.decoder(h)
        aux = {
            "latent": h,
            "flux_residual": torch.zeros(h.size(0), device=h.device, dtype=h.dtype),
        }
        return pred, aux
