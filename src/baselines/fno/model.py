from typing import Dict, Tuple

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """1D spectral convolution used by the FNO baseline."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        x_ft = torch.fft.rfft(x, dim=-1)
        num_modes = min(self.modes, x_ft.size(-1))
        out_ft = torch.zeros(
            x.size(0),
            self.out_channels,
            x_ft.size(-1),
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :num_modes] = torch.einsum(
            "bcm,com->bom",
            x_ft[:, :, :num_modes],
            self.weights[:, :, :num_modes],
        )
        out = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return out


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int, dropout: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.pointwise(x)
        y = self.act(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class GraphFNO(nn.Module):
    """FNO baseline on graphs by sorting nodes along a coordinate axis."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        modes: int = 16,
        num_layers: int = 4,
        dropout: float = 0.0,
        sort_axis: int = 0,
    ):
        super().__init__()
        self.width = width
        self.out_dim = out_dim
        self.sort_axis = sort_axis
        self.input_proj = nn.Conv1d(in_dim + 3, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes, dropout=dropout) for _ in range(num_layers)])
        self.output_proj = nn.Conv1d(width, out_dim, kernel_size=1)

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if batch.batch is None or batch.batch.numel() == 0:
            graph_ids = [0]
        else:
            num_graphs = batch.num_graphs if getattr(batch, "num_graphs", 0) else int(batch.batch.max().item()) + 1
            graph_ids = list(range(num_graphs))

        preds = []
        latents = []
        residuals = []
        for gid in graph_ids:
            if batch.batch is None or batch.batch.numel() == 0:
                idx = torch.arange(batch.x.size(0), device=batch.x.device)
            else:
                idx = (batch.batch == gid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            pos = batch.pos[idx]
            order = torch.argsort(pos[:, self.sort_axis])
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(order.numel(), device=order.device)

            features = torch.cat([batch.x[idx], pos], dim=-1)
            x_seq = features[order].transpose(0, 1).unsqueeze(0)
            h = self.input_proj(x_seq)
            for block in self.blocks:
                h = block(h)
            pred_seq = self.output_proj(h).squeeze(0).transpose(0, 1)
            pred = pred_seq[inv_order]

            latent_seq = h.squeeze(0).transpose(0, 1)
            latent = latent_seq[inv_order]

            preds.append(pred)
            latents.append(latent)
            residuals.append(torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype))

        if preds:
            pred_all = torch.cat(preds, dim=0)
            latent_all = torch.cat(latents, dim=0)
            residual_all = torch.cat(residuals, dim=0)
        else:
            device = batch.x.device
            dtype = batch.x.dtype
            pred_all = torch.zeros((0, self.out_dim), device=device, dtype=dtype)
            latent_all = torch.zeros((0, self.width), device=device, dtype=dtype)
            residual_all = torch.zeros((0,), device=device, dtype=dtype)

        aux = {
            "latent": latent_all,
            "flux_residual": residual_all,
        }
        return pred_all, aux
