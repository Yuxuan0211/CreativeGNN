#!/usr/bin/env python3
"""
Run inference with a trained Creative-GNN checkpoint and write predictions to CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import CFDDataset, build_dataloader
from src.models.creative_gnn import CreativeGNN
from train import _compute_stats, to_device


def _parse_fields(value: Optional[str], default: List[str]) -> List[str]:
    if not value:
        return default
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def _csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict flow fields with a trained Creative-GNN model.")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (model.pt).")
    parser.add_argument("--config", required=True, help="Path to config.yaml used for training.")
    parser.add_argument("--node-csv", required=True, help="Node CSV to predict.")
    parser.add_argument("--edge-csv", required=True, help="Edge CSV to use for the graph.")
    parser.add_argument("--out-csv", required=True, help="Output CSV for predictions.")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu).")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    in_fields = _parse_fields(
        cfg.get("in_fields"),
        [
            "Density__kg_m3_",
            "Eddy_Viscosity__Pa_s_",
            "Pressure__Pa_",
            "Static_Enthalpy__J_kg1_",
            "Temperature__K_",
        ],
    )
    out_fields = _parse_fields(
        cfg.get("out_fields"),
        [
            "Pressure__Pa_",
            "Velocity_u__m_s1_",
            "Velocity_v__m_s1_",
            "Velocity_w__m_s1_",
            "Density__kg_m3_",
        ],
    )

    dataset = CFDDataset(args.node_csv, args.edge_csv, in_fields=in_fields, out_fields=out_fields)
    loader = build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args") if isinstance(ckpt, dict) else None
    if ckpt_args:
        cfg.update(ckpt_args)
    state_dict = ckpt.get("model", ckpt)
    model = CreativeGNN(
        in_dim=cfg.get("in_dim", dataset.in_dim),
        out_dim=cfg.get("out_dim", dataset.out_dim),
        hidden_dim=cfg.get("hidden_dim", 32),
        num_regions=cfg.get("num_regions", 2),
        ode_steps=cfg.get("ode_steps", 1),
        ode_method=cfg.get("ode_method", "rk4"),
        ode_t=cfg.get("ode_t", 1.0),
        ode_rtol=cfg.get("ode_rtol"),
        ode_atol=cfg.get("ode_atol"),
        geom_type=cfg.get("geom_type", "rbf"),
        geom_dim=cfg.get("geom_dim", 16),
        geom_gamma=cfg.get("geom_gamma"),
        flux_dim=cfg.get("flux_dim", 5),
        flux_nproj=cfg.get("flux_nproj", 1),
        local_beta=cfg.get("local_beta", 1.0),
        local_depth=cfg.get("local_depth", 2),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    norm_stats = None
    if cfg.get("normalize", False):
        train_nodes = _csv_list(cfg.get("train_node_csv"))
        if train_nodes:
            train_edges = _csv_list(cfg.get("train_edge_csv"))
            if not train_edges:
                train_edges = [cfg.get("edge_csv", args.edge_csv)] * len(train_nodes)
            train_ds = CFDDataset(train_nodes, train_edges, in_fields=in_fields, out_fields=out_fields)
            mean_x, std_x, mean_y, std_y = _compute_stats(train_ds)
            norm_stats = (mean_x.to(device), std_x.to(device), mean_y.to(device), std_y.to(device))

    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            if norm_stats is not None:
                mean_x, std_x, mean_y, std_y = norm_stats
                batch.x = (batch.x - mean_x) / (std_x + cfg.get("norm_eps", 1e-6))
            pred, _ = model(batch, use_flux=not cfg.get("disable_flux", False))
            if norm_stats is not None:
                pred = pred * (std_y + cfg.get("norm_eps", 1e-6)) + mean_y
            preds.append(pred.cpu().numpy())

    pred_arr = preds[0]
    node_df = pd.read_csv(args.node_csv)
    for idx, field in enumerate(out_fields):
        node_df[f"Pred_{field}"] = pred_arr[:, idx]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    node_df.to_csv(args.out_csv, index=False)
    print(f"Wrote predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
