from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def supervised_loss(pred: torch.Tensor, target: torch.Tensor, volume: Optional[torch.Tensor] = None) -> torch.Tensor:
    if volume is None:
        return F.mse_loss(pred, target)
    # Weighted mean over nodes. Using .mean() here would divide by N twice when
    # volume is already normalized (e.g., volume_i = 1/N), making losses
    # artificially tiny and misleading.
    w = volume.unsqueeze(-1).clamp_min(0.0)
    err2 = (pred - target) ** 2
    return (w * err2).sum() / (w.sum() * err2.size(1) + 1e-12)


def flux_loss(residual: torch.Tensor, volume: Optional[torch.Tensor] = None) -> torch.Tensor:
    if residual.dim() == 1:
        per_node = residual.pow(2)
    else:
        per_node = residual.pow(2).sum(dim=-1)
    if volume is None:
        return per_node.mean()
    w = volume.clamp_min(0.0)
    return (w * per_node).sum() / (w.sum() + 1e-12)


def smoothness_loss(latent: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    diff = latent[src] - latent[dst]
    return (diff.pow(2).sum(dim=-1)).mean()


def vorticity_loss(
    pred: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    vel_indices: Optional[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    """Approximate vorticity via edge-based curl proxy."""
    if vel_indices is None:
        vel_indices = (1, 2, 3)
    if pred.size(1) <= max(vel_indices):
        return torch.tensor(0.0, device=pred.device)
    src, dst = edge_index
    vel = pred[:, vel_indices]
    vel_src, vel_dst = vel[src], vel[dst]
    edge_vec = pos[dst] - pos[src]
    edge_norm_sq = edge_vec.pow(2).sum(dim=-1, keepdim=True) + 1e-6
    curl_proxy = torch.cross(vel_dst - vel_src, edge_vec, dim=-1) / edge_norm_sq
    return curl_proxy.norm(dim=-1).mean()


def compute_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    volume: Optional[torch.Tensor],
    lambda_flux: float = 1.0,
    lambda_vort: float = 0.1,
    lambda_smooth: float = 0.1,
    vel_indices: Optional[Tuple[int, int, int]] = None,
    pred_denorm: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    losses["supervised"] = supervised_loss(pred, target, volume)
    flux_key = "flux_residual_post_norm" if "flux_residual_post_norm" in aux else "flux_residual_pre_norm"
    losses["flux"] = flux_loss(aux[flux_key], volume)
    vort_pred = pred_denorm if pred_denorm is not None else pred
    losses["vorticity"] = vorticity_loss(vort_pred, pos, edge_index, vel_indices=vel_indices)
    losses["smooth"] = smoothness_loss(aux["latent"], edge_index)
    losses["total"] = (
        losses["supervised"]
        + lambda_flux * losses["flux"]
        + lambda_vort * losses["vorticity"]
        + lambda_smooth * losses["smooth"]
    )
    return losses
