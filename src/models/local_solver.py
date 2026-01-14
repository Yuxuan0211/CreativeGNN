from typing import Optional

import torch
import torch.nn as nn

from .mlp import build_mlp


class RegionLocalSolver(nn.Module):
    """Lightweight region-adaptive operator to capture dominant local physics."""

    def __init__(self, hidden_dim: int, num_regions: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_regions = num_regions
        self.ops = nn.ModuleList(
            [
                build_mlp(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
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
        correction = torch.stack(outputs, dim=0).sum(dim=0)
        return self.norm(h + correction)
