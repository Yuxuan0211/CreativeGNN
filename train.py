import argparse
import csv
import glob
import json
import random
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import CFDDataset, SyntheticFlowDataset, build_dataloader, GraphBatch
from src.losses import compute_losses
from src.models.creative_gnn import CreativeGNN


def to_device(batch: GraphBatch, device: torch.device) -> GraphBatch:
    kwargs: Dict[str, torch.Tensor] = {}
    for field in ["x", "pos", "edge_index", "y", "region_mask", "volume", "batch"]:
        val = getattr(batch, field)
        kwargs[field] = val.to(device) if val is not None else None
    kwargs["num_graphs"] = batch.num_graphs
    return GraphBatch(**kwargs)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Creative-GNN on synthetic graphs or plug in CFD graphs.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-regions", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--ode-method", type=str, default="rk4")
    parser.add_argument("--ode-t", type=float, default=1.0)
    parser.add_argument("--ode-rtol", type=float, default=None)
    parser.add_argument("--ode-atol", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-graphs", type=int, default=64, help="Synthetic graphs for demo.")
    parser.add_argument("--in-dim", type=int, default=8)
    parser.add_argument("--out-dim", type=int, default=5)
    parser.add_argument("--lambda-flux", type=float, default=1.0)
    parser.add_argument("--lambda-vort", type=float, default=0.1)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--geom-type", type=str, default="rbf", choices=["rbf", "poly"])
    parser.add_argument("--geom-dim", type=int, default=16)
    parser.add_argument("--geom-gamma", type=float, default=None)
    parser.add_argument("--flux-dim", type=int, default=5)
    parser.add_argument("--flux-nproj", type=int, default=1)
    parser.add_argument("--local-beta", type=float, default=1.0)
    parser.add_argument("--local-depth", type=int, default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--loss-csv", type=str, default=None, help="Optional CSV path to log losses per epoch.")
    parser.add_argument("--vel-indices", type=str, default="1,2,3", help="Comma-separated velocity indices for vorticity loss.")
    parser.add_argument("--disable-flux", action="store_true", help="Disable flux layer during finetune/eval.")
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--eval-noise-std", type=float, default=0.0)
    parser.add_argument("--normalize", action="store_true", help="Normalize inputs/targets using train stats.")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA behavior.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory for logs/metrics.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to save a checkpoint.")
    parser.add_argument("--node-csv", type=str, default=None, help="Path to node feature CSV (e.g., Ma=2 t=0.1.csv).")
    parser.add_argument("--edge-csv", type=str, default=None, help="Path to edge triplet CSV (e.g., Desktop1_edges_triplet.csv).")
    parser.add_argument("--train-node-csv", type=str, default=None, help="Comma-separated or globbed train node CSVs.")
    parser.add_argument("--val-node-csv", type=str, default=None, help="Comma-separated or globbed val node CSVs.")
    parser.add_argument("--test-node-csv", type=str, default=None, help="Comma-separated or globbed test node CSVs.")
    parser.add_argument("--train-edge-csv", type=str, default=None, help="Train edge CSVs (defaults to --edge-csv).")
    parser.add_argument("--val-edge-csv", type=str, default=None, help="Val edge CSVs (defaults to --edge-csv).")
    parser.add_argument("--test-edge-csv", type=str, default=None, help="Test edge CSVs (defaults to --edge-csv).")
    parser.add_argument(
        "--in-fields",
        type=str,
        default=None,
        help="Comma-separated column names for inputs (default uses density, eddy viscosity, pressure, enthalpy, temperature).",
    )
    parser.add_argument(
        "--out-fields",
        type=str,
        default=None,
        help="Comma-separated column names for targets (default uses pressure + velocity u,v,w + density).",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file to override parameters.")
    return parser.parse_args(argv)


def _expand_paths(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        matches = glob.glob(p)
        out.extend(matches if matches else [p])
    return out


def _parse_indices(raw: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    indices = tuple(int(p) for p in parts)
    if len(indices) != 3:
        raise ValueError(f"vel-indices must have exactly 3 integers, got {indices}")
    return indices


def _ensure_loss_csv(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    fieldnames = ["epoch", "stage", "split", "lr", "supervised", "flux", "vorticity", "smooth", "total"]
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    return fieldnames


def _append_loss_csv(path: str, fieldnames: List[str], row: Dict[str, object]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_edge_paths(edge_csv: Optional[str], fallback: Optional[str], label: str) -> str:
    if edge_csv is not None:
        return edge_csv
    if fallback is not None:
        return fallback
    raise ValueError(f"Missing {label} edge CSV path.")


def _make_dataset(
    node_csvs: str,
    edge_csvs: str,
    in_fields: Optional[List[str]],
    out_fields: Optional[List[str]],
) -> CFDDataset:
    node_paths = _expand_paths(node_csvs)
    edge_paths = _expand_paths(edge_csvs)
    if len(edge_paths) == 1 and len(node_paths) > 1:
        edge_paths = edge_paths * len(node_paths)
    return CFDDataset(node_paths, edge_paths, in_fields=in_fields, out_fields=out_fields)


def _save_config(output_dir: Optional[str], args: argparse.Namespace) -> Optional[Path]:
    if output_dir is None:
        return None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=True)
    return cfg_path


def _log_line(output_dir: Optional[str], line: str) -> None:
    if output_dir is None:
        return
    log_path = Path(output_dir) / "train.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _compute_stats(dataset) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sum_x = None
    sum_y = None
    sum_x2 = None
    sum_y2 = None
    count = 0
    for sample in dataset:
        x = sample.x.float()
        y = sample.y.float()
        if sum_x is None:
            sum_x = x.sum(dim=0)
            sum_y = y.sum(dim=0)
            sum_x2 = (x * x).sum(dim=0)
            sum_y2 = (y * y).sum(dim=0)
        else:
            sum_x += x.sum(dim=0)
            sum_y += y.sum(dim=0)
            sum_x2 += (x * x).sum(dim=0)
            sum_y2 += (y * y).sum(dim=0)
        count += x.size(0)
    if count == 0:
        raise ValueError("Cannot compute stats on empty dataset.")
    mean_x = sum_x / count
    mean_y = sum_y / count
    var_x = sum_x2 / count - mean_x * mean_x
    var_y = sum_y2 / count - mean_y * mean_y
    std_x = torch.sqrt(torch.clamp(var_x, min=0.0))
    std_y = torch.sqrt(torch.clamp(var_y, min=0.0))
    return mean_x, std_x, mean_y, std_y


def _evaluate(
    model: CreativeGNN,
    loader,
    device: torch.device,
    use_flux: bool,
    noise_std: float = 0.0,
    norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    norm_eps: float = 1e-6,
) -> Dict[str, float]:
    model.eval()
    sum_sq_err = None
    sum_sq_true = None
    flux_pre_sum = 0.0
    flux_post_sum = 0.0
    num_nodes = 0
    num_graphs = 0
    time_total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            if norm_stats is not None:
                mean_x, std_x, mean_y, std_y = norm_stats
                batch.x = (batch.x - mean_x) / (std_x + norm_eps)
                batch.y = (batch.y - mean_y) / (std_y + norm_eps)
            if noise_std > 0:
                batch.x = batch.x + torch.randn_like(batch.x) * noise_std
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            pred, aux = model(batch, use_flux=use_flux, compute_flux_metrics=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_total += time.perf_counter() - start
            if norm_stats is not None:
                pred = pred * (std_y + norm_eps) + mean_y
                target = batch.y * (std_y + norm_eps) + mean_y
            else:
                target = batch.y
            err = pred - target
            sq_err = err.pow(2).sum(dim=0)
            sq_true = target.pow(2).sum(dim=0)
            if sum_sq_err is None:
                sum_sq_err = sq_err
                sum_sq_true = sq_true
            else:
                sum_sq_err += sq_err
                sum_sq_true += sq_true
            flux_pre = aux["flux_residual_pre_norm"].norm(dim=-1).sum().item()
            flux_post = aux["flux_residual_post_norm"].norm(dim=-1).sum().item()
            flux_pre_sum += flux_pre
            flux_post_sum += flux_post
            num_nodes += batch.y.size(0)
            num_graphs += batch.num_graphs

    if sum_sq_err is None or sum_sq_true is None:
        return {}
    denom = sum_sq_true.clamp_min(1e-12)
    l2_rel = torch.sqrt(sum_sq_err / denom)
    l2_rel_mean = l2_rel.mean().item()
    flux_pre_mean = flux_pre_sum / max(num_nodes, 1)
    flux_post_mean = flux_post_sum / max(num_nodes, 1)
    flux_drop = 0.0
    if flux_pre_mean > 0:
        flux_drop = 1.0 - flux_post_mean / flux_pre_mean

    return {
        "l2_rel": l2_rel.tolist(),
        "l2_rel_mean": l2_rel_mean,
        "flux_residual_mean": flux_pre_mean,
        "flux_residual_post_mean": flux_post_mean,
        "flux_residual_drop": flux_drop,
        "time_infer_s": time_total / max(num_graphs, 1),
    }

def main() -> None:
    args = parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                print(f"Warning: unknown config key '{k}' ignored.")
    train_from_args(args)


def train_from_args(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    _set_seed(args.seed, args.deterministic)
    vel_indices = _parse_indices(args.vel_indices)
    _save_config(args.output_dir, args)
    if isinstance(args.norm_eps, str):
        args.norm_eps = float(args.norm_eps)

    in_fields: Optional[List[str]] = args.in_fields.split(",") if args.in_fields else None
    out_fields: Optional[List[str]] = args.out_fields.split(",") if args.out_fields else None
    train_ds = None
    val_ds = None
    test_ds = None
    train_cases: List[str] = []
    val_cases: List[str] = []
    test_cases: List[str] = []

    if args.train_node_csv or args.val_node_csv or args.test_node_csv:
        if args.train_node_csv is None or args.val_node_csv is None:
            raise ValueError("train_node_csv and val_node_csv must be provided together.")
        train_edge = _resolve_edge_paths(args.train_edge_csv, args.edge_csv, "train")
        val_edge = _resolve_edge_paths(args.val_edge_csv, args.edge_csv, "val")
        train_ds = _make_dataset(args.train_node_csv, train_edge, in_fields, out_fields)
        val_ds = _make_dataset(args.val_node_csv, val_edge, in_fields, out_fields)
        if args.test_node_csv:
            test_edge = _resolve_edge_paths(args.test_edge_csv, args.edge_csv, "test")
            test_ds = _make_dataset(args.test_node_csv, test_edge, in_fields, out_fields)
        args.in_dim = train_ds.in_dim
        args.out_dim = train_ds.out_dim
        out_fields = train_ds.out_fields
        train_cases = [Path(p).name for p in train_ds.node_paths]
        val_cases = [Path(p).name for p in val_ds.node_paths]
        if test_ds is not None:
            test_cases = [Path(p).name for p in test_ds.node_paths]
    elif args.node_csv is not None and args.edge_csv is not None:
        dataset = _make_dataset(args.node_csv, args.edge_csv, in_fields, out_fields)
        args.in_dim = dataset.in_dim
        args.out_dim = dataset.out_dim
        out_fields = dataset.out_fields
        if len(dataset) > 1:
            n_val = max(1, int(0.1 * len(dataset)))
            n_train = len(dataset) - n_val
            generator = torch.Generator().manual_seed(args.seed)
            train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)
        else:
            train_ds = dataset
            val_ds = dataset
        train_cases = [Path(p).name for p in dataset.node_paths]
    else:
        dataset = SyntheticFlowDataset(num_graphs=args.num_graphs, in_dim=args.in_dim, out_dim=args.out_dim)
        if len(dataset) > 1:
            n_val = max(1, int(0.1 * len(dataset)))
            n_train = len(dataset) - n_val
            generator = torch.Generator().manual_seed(args.seed)
            train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)
        else:
            train_ds = dataset
            val_ds = dataset
        out_fields = dataset.out_fields

    train_loader = build_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_dataloader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = None
    if test_ds is not None:
        test_loader = build_dataloader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.output_dir is not None:
        _log_line(args.output_dir, f"seed: {args.seed}")
        if train_cases:
            _log_line(args.output_dir, f"train_cases: {train_cases}")
        if val_cases:
            _log_line(args.output_dir, f"val_cases: {val_cases}")
        if test_cases:
            _log_line(args.output_dir, f"test_cases: {test_cases}")

    norm_stats = None
    if args.normalize:
        mean_x, std_x, mean_y, std_y = _compute_stats(train_ds)
        norm_stats = (mean_x.to(device), std_x.to(device), mean_y.to(device), std_y.to(device))
        if args.output_dir is not None:
            _log_line(args.output_dir, f"normalize: true")
            _log_line(args.output_dir, f"norm_eps: {args.norm_eps}")

    model = CreativeGNN(
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        num_regions=args.num_regions,
        ode_steps=args.ode_steps,
        ode_method=args.ode_method,
        ode_t=args.ode_t,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        geom_type=args.geom_type,
        geom_dim=args.geom_dim,
        geom_gamma=args.geom_gamma,
        flux_dim=args.flux_dim,
        flux_nproj=args.flux_nproj,
        local_beta=args.local_beta,
        local_depth=args.local_depth,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    total_epochs = args.pretrain_epochs + args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs) if args.lr_schedule == "cosine" else None
    fieldnames = _ensure_loss_csv(args.loss_csv)

    def run_epoch(
        loader,
        train: bool,
        use_flux: bool,
        lambda_flux: float,
    ) -> Dict[str, float]:
        if train:
            model.train()
        else:
            model.eval()
        totals: Dict[str, float] = {"supervised": 0.0, "flux": 0.0, "vorticity": 0.0, "smooth": 0.0, "total": 0.0}
        count = 0
        for batch in loader:
            batch = to_device(batch, device)
            if norm_stats is not None:
                mean_x, std_x, mean_y, std_y = norm_stats
                batch.x = (batch.x - mean_x) / (std_x + args.norm_eps)
                batch.y = (batch.y - mean_y) / (std_y + args.norm_eps)
            if train and args.input_noise_std > 0:
                batch.x = batch.x + torch.randn_like(batch.x) * args.input_noise_std
            with torch.set_grad_enabled(train):
                pred, aux = model(batch, use_flux=use_flux)
                pred_denorm = None
                if norm_stats is not None:
                    _, _, mean_y, std_y = norm_stats
                    pred_denorm = pred * (std_y + args.norm_eps) + mean_y
                loss_dict = compute_losses(
                    pred,
                    batch.y,
                    aux,
                    batch.pos,
                    batch.edge_index,
                    batch.volume,
                    lambda_flux=lambda_flux,
                    lambda_vort=args.lambda_vort,
                    lambda_smooth=args.lambda_smooth,
                    vel_indices=vel_indices,
                    pred_denorm=pred_denorm,
                )
                if train:
                    optimizer.zero_grad()
                    loss_dict["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            for key in totals:
                totals[key] += loss_dict[key].item()
            count += 1
        if count == 0:
            return {k: 0.0 for k in totals}
        return {k: v / count for k, v in totals.items()}

    # Stage 1: region-adaptive pretraining (skip flux layer)
    for epoch in range(1, args.pretrain_epochs + 1):
        global_epoch = epoch
        train_losses = run_epoch(train_loader, train=True, use_flux=False, lambda_flux=0.0)
        val_losses = run_epoch(val_loader, train=False, use_flux=False, lambda_flux=0.0)
        lr = optimizer.param_groups[0]["lr"]
        line = (
            f"[pretrain] Epoch {global_epoch:03d} | train loss {train_losses['total']:.4f} | "
            f"val loss {val_losses['total']:.4f}"
        )
        print(line)
        _log_line(args.output_dir, line)
        if args.loss_csv and fieldnames:
            _append_loss_csv(
                args.loss_csv,
                fieldnames,
                {
                    "epoch": global_epoch,
                    "stage": "pretrain",
                    "split": "train",
                    "lr": lr,
                    **train_losses,
                },
            )
            _append_loss_csv(
                args.loss_csv,
                fieldnames,
                {
                    "epoch": global_epoch,
                    "stage": "pretrain",
                    "split": "val",
                    "lr": lr,
                    **val_losses,
                },
            )
        if scheduler is not None:
            scheduler.step()

    # Stage 2: full model fine-tuning
    best_val = float("inf")
    patience = 0
    use_flux = not args.disable_flux
    for epoch in range(1, args.epochs + 1):
        global_epoch = args.pretrain_epochs + epoch
        lambda_flux = args.lambda_flux if use_flux else 0.0
        train_losses = run_epoch(train_loader, train=True, use_flux=use_flux, lambda_flux=lambda_flux)
        val_losses = run_epoch(val_loader, train=False, use_flux=use_flux, lambda_flux=lambda_flux)
        lr = optimizer.param_groups[0]["lr"]
        line = f"Epoch {global_epoch:03d} | train loss {train_losses['total']:.4f} | val loss {val_losses['total']:.4f}"
        print(line)
        _log_line(args.output_dir, line)
        if args.loss_csv and fieldnames:
            _append_loss_csv(
                args.loss_csv,
                fieldnames,
                {
                    "epoch": global_epoch,
                    "stage": "finetune",
                    "split": "train",
                    "lr": lr,
                    **train_losses,
                },
            )
            _append_loss_csv(
                args.loss_csv,
                fieldnames,
                {
                    "epoch": global_epoch,
                    "stage": "finetune",
                    "split": "val",
                    "lr": lr,
                    **val_losses,
                },
            )
        if scheduler is not None:
            scheduler.step()

        if args.early_stop_patience > 0:
            if val_losses["total"] < best_val - 1e-6:
                best_val = val_losses["total"]
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    line = f"Early stopping triggered at epoch {global_epoch:03d}."
                    print(line)
                    _log_line(args.output_dir, line)
                    break

    def _inject_l2_fields(split_metrics: Dict[str, object]) -> None:
        l2_vals = split_metrics.pop("l2_rel", None)
        if l2_vals is None:
            return
        if out_fields is None:
            split_metrics["l2_rel"] = l2_vals
            return
        for name, val in zip(out_fields, l2_vals):
            split_metrics[f"l2_{name}"] = val

    metrics: Dict[str, object] = {
        "seed": args.seed,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "out_fields": out_fields,
        "normalize": args.normalize,
    }
    metrics["val"] = _evaluate(
        model,
        val_loader,
        device,
        use_flux=use_flux,
        norm_stats=norm_stats,
        norm_eps=args.norm_eps,
    )
    _inject_l2_fields(metrics["val"])
    if test_loader is not None:
        metrics["test"] = _evaluate(
            model,
            test_loader,
            device,
            use_flux=use_flux,
            norm_stats=norm_stats,
            norm_eps=args.norm_eps,
        )
        _inject_l2_fields(metrics["test"])
    if args.eval_noise_std > 0:
        metrics["eval_noise_std"] = args.eval_noise_std
        metrics["val_noisy"] = _evaluate(
            model,
            val_loader,
            device,
            use_flux=use_flux,
            noise_std=args.eval_noise_std,
            norm_stats=norm_stats,
            norm_eps=args.norm_eps,
        )
        _inject_l2_fields(metrics["val_noisy"])
        if test_loader is not None:
            metrics["test_noisy"] = _evaluate(
                model,
                test_loader,
                device,
                use_flux=use_flux,
                noise_std=args.eval_noise_std,
                norm_stats=norm_stats,
                norm_eps=args.norm_eps,
            )
            _inject_l2_fields(metrics["test_noisy"])

    if args.output_dir is not None:
        metrics_path = Path(args.output_dir) / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        _log_line(args.output_dir, f"Wrote metrics to {metrics_path}")

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
