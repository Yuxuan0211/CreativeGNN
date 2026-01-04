from typing import Optional

import torch
import torch.nn as nn

from .mlp import build_mlp


class RegionLocalSolver(nn.Module):
    """Lightweight region-adaptive operator to capture dominant local physics."""

    def __init__(
        self,
        hidden_dim: int,
        num_regions: int = 4,
        dropout: float = 0.0,
        beta: float = 1.0,
        depth: int = 2,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.beta = beta
        self.ops = nn.ModuleList(
            [
                build_mlp(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=depth, dropout=dropout)
                for _ in range(num_regions)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        rel = h[src] - h[dst]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, rel)

        if region_mask is None:
            region_mask = torch.full((h.size(0), self.num_regions), 1.0 / self.num_regions, device=h.device)
        if region_mask.shape[1] != self.num_regions:
            # Pad or truncate to expected number of regions
            r = region_mask.shape[1]
            if r < self.num_regions:
                pad = torch.zeros(h.size(0), self.num_regions - r, device=h.device)
                region_mask = torch.cat([region_mask, pad], dim=1)
            else:
                region_mask = region_mask[:, : self.num_regions]

        outputs = []
        for ridx, op in enumerate(self.ops):
            weight = region_mask[:, ridx].unsqueeze(-1)
            out = op(torch.cat([h, agg], dim=-1))
            outputs.append(out * weight)
        weights = region_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        h_star = torch.stack(outputs, dim=0).sum(dim=0) / weights
        h_next = h + self.beta * (h_star - h)
        return self.norm(h_next)
