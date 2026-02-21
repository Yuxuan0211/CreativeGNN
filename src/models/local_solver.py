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
        local_depth: int = 2,
        local_beta: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.local_depth = max(1, int(local_depth))
        self.local_beta = float(local_beta)

        self.ops = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        build_mlp(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
                        for _ in range(self.local_depth)
                    ]
                )
                for _ in range(num_regions)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.local_depth)])

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index

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

        for d in range(self.local_depth):
            rel = h[src] - h[dst]
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, rel)
            outputs = []
            for ridx, op_stack in enumerate(self.ops):
                weight = region_mask[:, ridx].unsqueeze(-1)
                out = op_stack[d](torch.cat([h, agg], dim=-1))
                outputs.append(out * weight)
            correction = torch.stack(outputs, dim=0).sum(dim=0)
            h = self.norms[d](h + self.local_beta * correction)
        return h
