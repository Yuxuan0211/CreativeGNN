import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import CFDDataset, build_dataloader
from src.models.creative_gnn import CreativeGNN
from train import _compute_stats


def _expand_paths(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out = []
    import glob

    for p in parts:
        matches = glob.glob(p)
        out.extend(matches if matches else [p])
    return out


def estimate_freestream(y_np: np.ndarray, region_mask: Optional[np.ndarray]) -> dict:
    # y columns: [Pressure, U, V, W, Density]
    p = y_np[:, 0]
    u = y_np[:, 1]
    v = y_np[:, 2]
    w = y_np[:, 3]
    rho = y_np[:, 4]
    speed = np.sqrt(u * u + v * v + w * w)
    if region_mask is not None:
        free_w = region_mask[:, -1]
        mask = free_w > 0.9
        if mask.sum() < 10:
            mask = free_w >= np.percentile(free_w, 75.0)
    else:
        # fallback: use 5th--95th percentile trimming to pick bulk freestream
        mask = (np.abs(p - np.median(p)) < 0.05 * np.ptp(p))
        if mask.sum() < 10:
            mask = np.ones_like(p, dtype=bool)
    p_inf = float(np.median(p[mask]))
    rho_inf = float(np.median(rho[mask]))
    V_inf = float(np.median(speed[mask]))
    return {"p_inf": p_inf, "rho_inf": rho_inf, "V_inf": V_inf}


def compute_cp(p: np.ndarray, p_inf: float, rho_inf: float, V_inf: float) -> np.ndarray:
    denom = 0.5 * rho_inf * (V_inf ** 2 + 1e-12)
    return (p - p_inf) / denom


def make_model_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module:
    saved_args = ckpt.get("args", {})
    model_args = {}
    # required constructor fields
    for k in ["in_dim", "out_dim", "hidden_dim", "num_regions", "ode_steps", "ode_method", "ode_t", "ode_rtol", "ode_atol", "geom_type", "geom_dim", "geom_gamma", "flux_dim", "flux_nproj", "local_beta", "local_depth"]:
        if k in saved_args:
            model_args[k] = saved_args[k]
    model = CreativeGNN(
        in_dim=model_args.get("in_dim", 5),
        out_dim=model_args.get("out_dim", 5),
        hidden_dim=model_args.get("hidden_dim", 128),
        num_regions=model_args.get("num_regions", 4),
        ode_steps=model_args.get("ode_steps", 4),
        ode_method=model_args.get("ode_method", "rk4"),
        ode_t=model_args.get("ode_t", 1.0),
        ode_rtol=model_args.get("ode_rtol", None),
        ode_atol=model_args.get("ode_atol", None),
        geom_type=model_args.get("geom_type", "rbf"),
        geom_dim=model_args.get("geom_dim", 16),
        geom_gamma=model_args.get("geom_gamma", None),
        flux_dim=model_args.get("flux_dim", 5),
        flux_nproj=model_args.get("flux_nproj", 3),
        local_beta=model_args.get("local_beta", 1.0),
        local_depth=model_args.get("local_depth", 2),
    ).to(device)
    return model


def run_evaluation(args):
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model = make_model_from_ckpt(ckpt, device)
    state = ckpt.get("model") or ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    test_node_paths = _expand_paths(args.test_node_csv)
    test_edge_paths = _expand_paths(args.test_edge_csv) if args.test_edge_csv else []
    if len(test_edge_paths) == 0 and len(test_node_paths) > 0:
        # try to reuse the same edge file for all nodes by searching _edge_cache
        # user should supply edge CSVs; if not, CFDDataset will raise.
        pass

    dataset = CFDDataset(test_node_paths, test_edge_paths)
    loader = build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)
    out_dir = Path(args.output_dir) if args.output_dir else Path("eval_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_stats = None
    if ckpt.get("args", {}).get("normalize", False):
        train_node_paths = _expand_paths(ckpt.get("args", {}).get("train_node_csv"))
        train_edge_paths = _expand_paths(ckpt.get("args", {}).get("train_edge_csv"))
        if train_node_paths:
            if len(train_edge_paths) == 1 and len(train_node_paths) > 1:
                train_edge_paths = train_edge_paths * len(train_node_paths)
            train_ds = CFDDataset(train_node_paths, train_edge_paths)
            mean_x, std_x, mean_y, std_y = _compute_stats(train_ds)
            norm_stats = (
                mean_x.to(device),
                std_x.to(device),
                mean_y.to(device),
                std_y.to(device),
            )

    results = {"cases": []}
    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            # single graph
            x = sample.x.to(device)
            pos = sample.pos.to(device)
            edge_index = sample.edge_index.to(device)
            y_true = sample.y.to(device)
            region_mask = sample.region_mask.numpy() if sample.region_mask is not None else None
            batch = type("B", (), {"x": x.unsqueeze(0), "pos": pos.unsqueeze(0), "edge_index": edge_index.unsqueeze(0), "y": y_true.unsqueeze(0), "region_mask": sample.region_mask, "volume": sample.volume, "batch": torch.zeros(x.size(0), dtype=torch.long), "num_graphs": 1})
            # But model expects GraphBatch with flat tensors; reuse train.to_device is internal; instead call model with a minimal GraphBatch-like object
            # Construct GraphBatch-like
            from src.data import GraphBatch
            gb = GraphBatch(x=sample.x, pos=sample.pos, edge_index=sample.edge_index, y=sample.y, region_mask=sample.region_mask, volume=sample.volume, batch=torch.zeros(sample.x.size(0), dtype=torch.long), num_graphs=1)
            gb = gb
            gb = gb
            gb = type("GB", (), {})()
            setattr(gb, "x", sample.x.to(device))
            setattr(gb, "pos", sample.pos.to(device))
            setattr(gb, "edge_index", sample.edge_index.to(device))
            setattr(gb, "y", sample.y.to(device))
            setattr(gb, "region_mask", sample.region_mask.to(device) if sample.region_mask is not None else None)
            setattr(gb, "volume", sample.volume.to(device) if sample.volume is not None else None)
            setattr(gb, "batch", torch.zeros(sample.x.size(0), dtype=torch.long).to(device))
            setattr(gb, "num_graphs", 1)

            if norm_stats is not None:
                mean_x, std_x, _, _ = norm_stats
                gb.x = (gb.x - mean_x) / (std_x + 1e-6)

            pred, aux = model(gb, use_flux=not args.disable_flux)
            if norm_stats is not None:
                _, _, mean_y, std_y = norm_stats
                pred = pred * (std_y + 1e-6) + mean_y
            pred_np = pred.cpu().numpy()
            true_np = sample.y.numpy()
            pos_np = sample.pos.numpy()

            stats = estimate_freestream(true_np, region_mask)
            p_inf = stats["p_inf"]
            rho_inf = stats["rho_inf"]
            V_inf = stats["V_inf"]
            cp_true = compute_cp(true_np[:, 0], p_inf, rho_inf, V_inf)
            cp_pred = compute_cp(pred_np[:, 0], p_inf, rho_inf, V_inf)

            # Line plot Cp vs X-coordinate styled to match reference loss_curves.png
            # Reference image size: 2720 x 1497 px; use similar figsize at 200 DPI
            ref_w_px, ref_h_px = 2720, 1497
            dpi = 200
            figsize = (ref_w_px / dpi, ref_h_px / dpi)
            # sort by x for a smooth curve
            order = np.argsort(pos_np[:, 0])
            x_sorted = pos_np[order, 0]
            cp_true_s = cp_true[order]
            cp_pred_s = cp_pred[order]

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            # background and grid similar to loss curves
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

            # plot lines with stronger width
            ax.plot(x_sorted, cp_true_s, color="#1f77b4", linewidth=2.2, label="true")
            ax.plot(x_sorted, cp_pred_s, color="#ff7f0e", linewidth=2.2, linestyle=(0, (5, 2)), label="pred")
            ax.scatter(x_sorted, cp_true_s, s=10, color="#1f77b4", alpha=0.6)
            ax.scatter(x_sorted, cp_pred_s, s=10, color="#ff7f0e", alpha=0.6)

            ax.set_xlabel("X (m)", fontsize=18)
            ax.set_ylabel("Cp", fontsize=18)
            ax.set_title(f"Case {Path(dataset.node_paths[idx]).name}", fontsize=20)
            ax.tick_params(axis="both", which="major", labelsize=14)
            legend = ax.legend(fontsize=16)
            legend.get_frame().set_alpha(0.9)

            fig_path = out_dir / f"cp_compare_{idx}_{Path(dataset.node_paths[idx]).name}.png"
            fig.savefig(str(fig_path), dpi=dpi, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)

            npz_path = out_dir / f"cp_vals_{idx}_{Path(dataset.node_paths[idx]).name}.npz"
            np.savez_compressed(str(npz_path), cp_true=cp_true, cp_pred=cp_pred, x=pos_np[:, 0])

            results["cases"].append({
                "case": Path(dataset.node_paths[idx]).name,
                "cp_plot": str(fig_path),
                "cp_values": str(npz_path),
                "p_inf": p_inf,
                "rho_inf": rho_inf,
                "V_inf": V_inf,
            })

    results_path = out_dir / "eval_summary.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote evaluation outputs to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test-node-csv", required=True, help="Comma-separated test node CSVs or glob")
    parser.add_argument("--test-edge-csv", default=None, help="Comma-separated test edge CSVs or glob (optional)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="eval_out")
    parser.add_argument("--disable-flux", action="store_true")
    args = parser.parse_args()
    run_evaluation(args)
