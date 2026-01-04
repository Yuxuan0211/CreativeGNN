#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader


RUN_ROOT = Path("/root/autodl-tmp/CreativeGNN/exp_runs/20260103_140137")
PDF_PATH = Path("/root/autodl-tmp/CreativeGNN/CreativeGNN_FigurePack_Expanded_v4.pdf")
OUT_DIR = RUN_ROOT / "summary_png_20260104_164152"


def value_to_float(value: str) -> float | None:
    if value is None:
        return None
    s = str(value)
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    if re.fullmatch(r"-?\d+p\d+", s):
        return float(s.replace("p", "."))
    return None


def parse_losses(run_root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(run_root.rglob("losses.csv")):
        rel = path.relative_to(run_root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        module, param, value = parts[0], parts[1], parts[2]
        seed = parts[-2]
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df = df.copy()
        df["module"] = module
        df["param"] = param
        df["value"] = value
        df["seed"] = seed
        records.append(df)
    if not records:
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    df["seed"] = df["seed"].astype(str)
    return df


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    df["seed"] = df["seed"].astype(str)
    return df


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_sensitivity(df: pd.DataFrame, module: str, param: str, metric: str, out_path: Path, title: str, logx: bool = False) -> None:
    g = df[(df["module"] == module) & (df["param"] == param)]
    g = g.dropna(subset=[metric])
    if g.empty:
        return
    g = g.copy()
    g["x"] = g["value_num"].where(g["value_num"].notna(), g["value"].astype(str))
    agg = g.groupby("x")[metric].agg(["mean", "std"]).reset_index().sort_values("x")
    x = agg["x"].values
    y = agg["mean"].values
    yerr = agg["std"].fillna(0).values

    plt.figure(figsize=(6.5, 4.2))
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, lw=1.5)
    plt.title(title)
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    if not np.issubdtype(agg["x"].dtype, np.number):
        plt.xticks(range(len(x)), [str(v) for v in x], rotation=30, ha="right")
    if logx:
        plt.xscale("symlog", linthresh=1e-4)
    save_fig(out_path)


def last_epoch_stats(losses: pd.DataFrame, split: str, stage: str, comp: str) -> pd.DataFrame:
    g = losses[(losses["split"] == split) & (losses["stage"] == stage)]
    if g.empty:
        return pd.DataFrame()
    g = g.sort_values("epoch")
    last = g.groupby(["module", "param", "value", "seed"]).tail(1)
    return last[["module", "param", "value", "seed", comp, "epoch"]]


def tail_mean(losses: pd.DataFrame, split: str, stage: str, comp: str, tail: int = 10) -> pd.DataFrame:
    g = losses[(losses["split"] == split) & (losses["stage"] == stage)]
    if g.empty:
        return pd.DataFrame()

    def _tail_mean(df: pd.DataFrame) -> float:
        return df.sort_values("epoch")[comp].tail(tail).mean()

    res = g.groupby(["module", "param", "value", "seed"]).apply(_tail_mean).reset_index(name=comp)
    return res


def auc_metric(losses: pd.DataFrame, split: str, stage: str, comp: str) -> pd.DataFrame:
    g = losses[(losses["split"] == split) & (losses["stage"] == stage)]
    if g.empty:
        return pd.DataFrame()

    def _auc(df: pd.DataFrame) -> float:
        df = df.sort_values("epoch")
        return float(np.trapz(df[comp].values, df["epoch"].values))

    res = g.groupby(["module", "param", "value", "seed"]).apply(_auc).reset_index(name=comp)
    return res


def convergence_epoch(losses: pd.DataFrame, split: str, stage: str, comp: str, pct: float = 0.05) -> pd.DataFrame:
    g = losses[(losses["split"] == split) & (losses["stage"] == stage)]
    if g.empty:
        return pd.DataFrame()

    def _conv(df: pd.DataFrame) -> int:
        df = df.sort_values("epoch")
        min_val = df[comp].min()
        thresh = min_val * (1.0 + pct)
        idx = df[df[comp] <= thresh]
        if idx.empty:
            return int(df["epoch"].iloc[-1])
        return int(idx["epoch"].iloc[0])

    res = g.groupby(["module", "param", "value", "seed"]).apply(_conv).reset_index(name="convergence_epoch")
    return res


def plot_comp_metric(stats: pd.DataFrame, module: str, param: str, comp: str, out_path: Path, title: str, logx: bool = False) -> None:
    g = stats[(stats["module"] == module) & (stats["param"] == param)]
    if g.empty:
        return
    g = g.copy()
    g["value_num"] = g["value"].map(value_to_float)
    g["x"] = g["value_num"].where(g["value_num"].notna(), g["value"].astype(str))
    agg = g.groupby("x")[comp].agg(["mean", "std"]).reset_index().sort_values("x")
    x = agg["x"].values
    y = agg["mean"].values
    yerr = agg["std"].fillna(0).values

    plt.figure(figsize=(6.5, 4.2))
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, lw=1.5)
    plt.title(title)
    plt.xlabel(param)
    plt.ylabel(comp)
    plt.grid(True, alpha=0.3)
    if not np.issubdtype(agg["x"].dtype, np.number):
        plt.xticks(range(len(x)), [str(v) for v in x], rotation=30, ha="right")
    if logx:
        plt.xscale("symlog", linthresh=1e-4)
    save_fig(out_path)


def plot_curves(losses: pd.DataFrame, module: str, param: str, comp: str, out_path: Path, title: str) -> None:
    g = losses[(losses["module"] == module) & (losses["param"] == param) & (losses["stage"] == "finetune") & (losses["split"] == "val")]
    if g.empty:
        return
    g = g.copy()
    g["value_num"] = g["value"].map(value_to_float)
    g["label"] = g["value_num"].where(g["value_num"].notna(), g["value"].astype(str))

    plt.figure(figsize=(6.8, 4.6))
    for value, dfv in g.groupby("label"):
        curve = dfv.groupby("epoch")[comp].mean().reset_index()
        plt.plot(curve["epoch"], curve[comp], label=str(value))
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(comp)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=7)
    save_fig(out_path)


def plot_heat(losses: pd.DataFrame, module: str, param: str, comp: str, out_path: Path, title: str) -> None:
    g = losses[(losses["module"] == module) & (losses["param"] == param) & (losses["stage"] == "finetune") & (losses["split"] == "val")]
    if g.empty:
        return
    g = g.copy()
    g["value_num"] = g["value"].map(value_to_float)
    g["label"] = g["value_num"].where(g["value_num"].notna(), g["value"].astype(str))
    pivot = g.groupby(["epoch", "label"])[comp].mean().reset_index().pivot(index="epoch", columns="label", values=comp)
    pivot = pivot.sort_index(axis=1)
    data = pivot.values

    plt.figure(figsize=(6.5, 4.8))
    plt.imshow(data, aspect="auto", cmap="viridis", origin="lower")
    plt.title(title)
    plt.xlabel(param)
    plt.ylabel("epoch")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=30, ha="right")
    plt.colorbar(label=comp)
    save_fig(out_path)


def plot_ablation(summary: pd.DataFrame, module: str, param: str, metric: str, out_path: Path, title: str) -> None:
    g = summary[(summary["module"] == module) & (summary["param"] == param)]
    g = g.dropna(subset=[metric])
    if g.empty:
        return
    g = g.copy()
    g["label"] = g["value"].astype(str)
    agg = g.groupby("label")[metric].agg(["mean", "std"]).reset_index().sort_values("label")
    x = np.arange(len(agg))
    plt.figure(figsize=(5.8, 4.0))
    plt.bar(x, agg["mean"], yerr=agg["std"].fillna(0), capsize=4, color="#4C78A8")
    plt.xticks(x, agg["label"], rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(metric)
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)


def plot_box(summary: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    g = summary.dropna(subset=[metric])
    if g.empty:
        return
    groups = [df[metric].values for _, df in g.groupby("module")]
    labels = [m for m, _ in g.groupby("module")]
    plt.figure(figsize=(7.0, 4.4))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)


def plot_overview_dot(summary: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    g = summary.dropna(subset=[metric]).copy()
    if g.empty:
        return
    g["label"] = g.apply(lambda r: f"{r['module']}/{r['param']}={r['value']}/seed{r['seed']}", axis=1)
    g = g.sort_values(metric)
    plt.figure(figsize=(8.0, max(4.0, 0.08 * len(g))))
    plt.scatter(g[metric], np.arange(len(g)), s=12, alpha=0.7)
    plt.yticks(np.arange(len(g)), g["label"], fontsize=6)
    plt.title(title)
    plt.xlabel(metric)
    plt.grid(True, axis="x", alpha=0.3)
    save_fig(out_path)


def plot_pareto(summary: pd.DataFrame, out_path: Path) -> None:
    g = summary.dropna(subset=["test_l2_rel_mean", "test_flux_residual_mean"]).copy()
    if g.empty:
        return
    plt.figure(figsize=(6.4, 4.6))
    plt.scatter(g["test_l2_rel_mean"], g["test_flux_residual_mean"], s=14, alpha=0.7)
    plt.title("Test L2 vs Test Flux Residual")
    plt.xlabel("test_l2_rel_mean")
    plt.ylabel("test_flux_residual_mean")
    plt.grid(True, alpha=0.3)
    save_fig(out_path)


def plot_top(summary: pd.DataFrame, metric: str, out_path: Path, title: str, topn: int = 10) -> None:
    g = summary.dropna(subset=[metric]).copy()
    if g.empty:
        return
    g = g.sort_values(metric).head(topn)
    labels = g.apply(lambda r: f"{r['module']}/{r['param']}={r['value']}/seed{r['seed']}", axis=1)
    x = np.arange(len(g))
    plt.figure(figsize=(7.5, 4.4))
    plt.bar(x, g[metric], color="#59A14F")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=6)
    plt.title(title)
    plt.ylabel(metric)
    plt.grid(True, axis="y", alpha=0.3)
    save_fig(out_path)


def plot_corr(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    g = df.dropna(subset=[x, y])
    if g.empty:
        return
    plt.figure(figsize=(6.0, 4.2))
    plt.scatter(g[x], g[y], s=14, alpha=0.7)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.3)
    save_fig(out_path)


def plot_delta(summary: pd.DataFrame, module: str, param: str, metric: str, ref_value: str, out_path: Path, title: str, logx: bool = False) -> None:
    g = summary[(summary["module"] == module) & (summary["param"] == param)].dropna(subset=[metric]).copy()
    if g.empty:
        return
    g["value_num"] = g["value"].map(value_to_float)
    ref = g[g["value"].astype(str) == ref_value]
    if ref.empty:
        return
    ref_mean = ref[metric].mean()
    agg = g.groupby("value")[metric].mean().reset_index()
    agg["delta"] = agg[metric] - ref_mean
    agg["x"] = agg["value"].map(value_to_float).where(pd.notna(agg["value"].map(value_to_float)), agg["value"].astype(str))
    agg = agg.sort_values("x")
    x = agg["x"].values
    y = agg["delta"].values
    plt.figure(figsize=(6.3, 4.1))
    plt.plot(x, y, "o-", lw=1.5)
    plt.axhline(0.0, color="black", lw=1, alpha=0.6)
    plt.title(title)
    plt.xlabel(param)
    plt.ylabel(f"delta {metric} (vs {ref_value})")
    plt.grid(True, alpha=0.3)
    if not np.issubdtype(agg["x"].dtype, np.number):
        plt.xticks(range(len(x)), [str(v) for v in x], rotation=30, ha="right")
    if logx:
        plt.xscale("symlog", linthresh=1e-4)
    save_fig(out_path)


def main() -> None:
    ensure_dir(OUT_DIR)
    summary = load_summary(RUN_ROOT / "summary.csv")
    losses = parse_losses(RUN_ROOT)

    # Sensitivity (test metrics)
    plot_sensitivity(summary, "loss", "lambda_flux", "test_l2_rel_mean", OUT_DIR / "sens_lambda_testl2.png", "lambda_flux vs Test L2", logx=True)
    plot_sensitivity(summary, "loss", "lambda_flux", "test_flux_residual_mean", OUT_DIR / "sens_lambda_testflux.png", "lambda_flux vs Test Flux Residual", logx=True)
    plot_sensitivity(summary, "loss", "lambda_flux", "test_flux_residual_post_mean", OUT_DIR / "sens_lambda_testflux_post.png", "lambda_flux vs Test Flux Residual (post)", logx=True)
    plot_sensitivity(summary, "local_solver", "beta", "test_l2_rel_mean", OUT_DIR / "sens_beta_testl2.png", "beta vs Test L2")
    plot_sensitivity(summary, "local_solver", "beta", "test_flux_residual_mean", OUT_DIR / "sens_beta_testflux.png", "beta vs Test Flux Residual")
    plot_sensitivity(summary, "message_passing", "k_hop", "test_l2_rel_mean", OUT_DIR / "sens_khop_testl2.png", "k-hop vs Test L2")
    plot_sensitivity(summary, "message_passing", "k_hop", "test_flux_residual_mean", OUT_DIR / "sens_khop_testflux.png", "k-hop vs Test Flux Residual")
    plot_sensitivity(summary, "flux_layer", "nproj", "test_l2_rel_mean", OUT_DIR / "sens_nproj_testl2.png", "nproj vs Test L2")
    plot_sensitivity(summary, "flux_layer", "nproj", "test_flux_residual_mean", OUT_DIR / "sens_nproj_testflux.png", "nproj vs Test Flux Residual")

    # Sensitivity using train flux loss (tail mean)
    tail_train_flux = tail_mean(losses, split="train", stage="finetune", comp="flux")
    plot_comp_metric(tail_train_flux, "loss", "lambda_flux", "flux", OUT_DIR / "sens_lambda_trainfluxloss.png", "lambda_flux vs Train Flux Loss (tail)", logx=True)
    plot_comp_metric(tail_train_flux, "local_solver", "beta", "flux", OUT_DIR / "sens_beta_trainfluxloss.png", "beta vs Train Flux Loss (tail)")
    plot_comp_metric(tail_train_flux, "message_passing", "k_hop", "flux", OUT_DIR / "sens_khop_trainfluxloss.png", "k-hop vs Train Flux Loss (tail)")
    plot_comp_metric(tail_train_flux, "flux_layer", "nproj", "flux", OUT_DIR / "sens_nproj_trainfluxloss.png", "nproj vs Train Flux Loss (tail)")

    # Sensitivity convergence (5% epoch on val flux)
    conv = convergence_epoch(losses, split="val", stage="finetune", comp="flux", pct=0.05)
    plot_comp_metric(conv, "loss", "lambda_flux", "convergence_epoch", OUT_DIR / "sens_lambda_convergence.png", "lambda_flux vs Convergence Epoch", logx=True)
    plot_comp_metric(conv, "local_solver", "beta", "convergence_epoch", OUT_DIR / "sens_beta_convergence.png", "beta vs Convergence Epoch")
    plot_comp_metric(conv, "message_passing", "k_hop", "convergence_epoch", OUT_DIR / "sens_khop_convergence.png", "k-hop vs Convergence Epoch")

    # Curves (val loss curves)
    plot_curves(losses, "loss", "lambda_flux", "flux", OUT_DIR / "curves_lambda_flux.png", "lambda_flux: val flux loss curves")
    plot_curves(losses, "loss", "lambda_flux", "total", OUT_DIR / "curves_lambda_total.png", "lambda_flux: val total loss curves")
    plot_curves(losses, "local_solver", "beta", "flux", OUT_DIR / "curves_beta_flux.png", "beta: val flux loss curves")
    plot_curves(losses, "local_solver", "beta", "total", OUT_DIR / "curves_beta_total.png", "beta: val total loss curves")
    plot_curves(losses, "message_passing", "k_hop", "flux", OUT_DIR / "curves_khop_flux.png", "k-hop: val flux loss curves")
    plot_curves(losses, "message_passing", "k_hop", "total", OUT_DIR / "curves_khop_total.png", "k-hop: val total loss curves")
    plot_curves(losses, "flux_layer", "nproj", "flux", OUT_DIR / "curves_nproj_flux.png", "nproj: val flux loss curves")
    plot_curves(losses, "flux_layer", "use_flux", "flux", OUT_DIR / "curves_useflux_flux.png", "use_flux: val flux loss curves")
    plot_curves(losses, "flux_layer", "use_flux", "supervised", OUT_DIR / "curves_useflux_supervised.png", "use_flux: val supervised loss curves")
    plot_curves(losses, "flux_layer", "use_flux", "total", OUT_DIR / "curves_useflux_total.png", "use_flux: val total loss curves")
    plot_curves(losses, "train", "curriculum", "flux", OUT_DIR / "curves_curriculum_flux.png", "curriculum: val flux loss curves")
    plot_curves(losses, "train", "curriculum", "total", OUT_DIR / "curves_curriculum_total.png", "curriculum: val total loss curves")

    # Component comparison at last epoch (val)
    last_val = last_epoch_stats(losses, split="val", stage="finetune", comp="total")
    last_val_flux = last_epoch_stats(losses, split="val", stage="finetune", comp="flux")
    last_val_sup = last_epoch_stats(losses, split="val", stage="finetune", comp="supervised")
    last_val_vort = last_epoch_stats(losses, split="val", stage="finetune", comp="vorticity")
    last_val_smooth = last_epoch_stats(losses, split="val", stage="finetune", comp="smooth")

    plot_comp_metric(last_val_flux, "loss", "lambda_flux", "flux", OUT_DIR / "lambda_comp_flux.png", "lambda_flux vs val flux (last)", logx=True)
    plot_comp_metric(last_val_sup, "loss", "lambda_flux", "supervised", OUT_DIR / "lambda_comp_supervised.png", "lambda_flux vs val supervised (last)", logx=True)
    plot_comp_metric(last_val_vort, "loss", "lambda_flux", "vorticity", OUT_DIR / "lambda_comp_vorticity.png", "lambda_flux vs val vorticity (last)", logx=True)
    plot_comp_metric(last_val_smooth, "loss", "lambda_flux", "smooth", OUT_DIR / "lambda_comp_smooth.png", "lambda_flux vs val smooth (last)", logx=True)
    plot_comp_metric(last_val, "loss", "lambda_flux", "total", OUT_DIR / "lambda_comp_total.png", "lambda_flux vs val total (last)", logx=True)

    plot_comp_metric(last_val_flux, "local_solver", "beta", "flux", OUT_DIR / "beta_comp_flux.png", "beta vs val flux (last)")
    plot_comp_metric(last_val_sup, "local_solver", "beta", "supervised", OUT_DIR / "beta_comp_supervised.png", "beta vs val supervised (last)")
    plot_comp_metric(last_val_vort, "local_solver", "beta", "vorticity", OUT_DIR / "beta_comp_vorticity.png", "beta vs val vorticity (last)")
    plot_comp_metric(last_val_smooth, "local_solver", "beta", "smooth", OUT_DIR / "beta_comp_smooth.png", "beta vs val smooth (last)")
    plot_comp_metric(last_val, "local_solver", "beta", "total", OUT_DIR / "beta_comp_total.png", "beta vs val total (last)")

    plot_comp_metric(last_val_flux, "message_passing", "k_hop", "flux", OUT_DIR / "khop_comp_flux.png", "k-hop vs val flux (last)")
    plot_comp_metric(last_val_sup, "message_passing", "k_hop", "supervised", OUT_DIR / "khop_comp_supervised.png", "k-hop vs val supervised (last)")
    plot_comp_metric(last_val_vort, "message_passing", "k_hop", "vorticity", OUT_DIR / "khop_comp_vorticity.png", "k-hop vs val vorticity (last)")
    plot_comp_metric(last_val_smooth, "message_passing", "k_hop", "smooth", OUT_DIR / "khop_comp_smooth.png", "k-hop vs val smooth (last)")
    plot_comp_metric(last_val, "message_passing", "k_hop", "total", OUT_DIR / "khop_comp_total.png", "k-hop vs val total (last)")

    # Ratio flux/total at last epoch
    ratio = last_val_flux.merge(last_val, on=["module", "param", "value", "seed"], suffixes=("_flux", "_total"))
    ratio["ratio"] = ratio["flux"] / ratio["total"]
    plot_comp_metric(ratio, "loss", "lambda_flux", "ratio", OUT_DIR / "lambda_ratio_flux_over_total.png", "lambda_flux vs flux/total (last)", logx=True)
    plot_comp_metric(ratio, "local_solver", "beta", "ratio", OUT_DIR / "beta_ratio_flux_over_total.png", "beta vs flux/total (last)")
    plot_comp_metric(ratio, "message_passing", "k_hop", "ratio", OUT_DIR / "khop_ratio_flux_over_total.png", "k-hop vs flux/total (last)")

    # Tail and AUC metrics (val)
    tail_val_flux = tail_mean(losses, split="val", stage="finetune", comp="flux")
    tail_val_total = tail_mean(losses, split="val", stage="finetune", comp="total")
    auc_val_flux = auc_metric(losses, split="val", stage="finetune", comp="flux")
    auc_val_total = auc_metric(losses, split="val", stage="finetune", comp="total")

    plot_comp_metric(tail_val_flux, "loss", "lambda_flux", "flux", OUT_DIR / "lambda_tail_val_flux.png", "lambda_flux vs val flux (tail)", logx=True)
    plot_comp_metric(tail_val_total, "loss", "lambda_flux", "total", OUT_DIR / "lambda_tail_val_total.png", "lambda_flux vs val total (tail)", logx=True)
    plot_comp_metric(auc_val_flux, "loss", "lambda_flux", "flux", OUT_DIR / "lambda_auc_val_flux.png", "lambda_flux vs val flux AUC", logx=True)
    plot_comp_metric(auc_val_total, "loss", "lambda_flux", "total", OUT_DIR / "lambda_auc_val_total.png", "lambda_flux vs val total AUC", logx=True)

    plot_comp_metric(tail_val_flux, "local_solver", "beta", "flux", OUT_DIR / "beta_tail_val_flux.png", "beta vs val flux (tail)")
    plot_comp_metric(tail_val_total, "local_solver", "beta", "total", OUT_DIR / "beta_tail_val_total.png", "beta vs val total (tail)")
    plot_comp_metric(auc_val_flux, "local_solver", "beta", "flux", OUT_DIR / "beta_auc_val_flux.png", "beta vs val flux AUC")
    plot_comp_metric(auc_val_total, "local_solver", "beta", "total", OUT_DIR / "beta_auc_val_total.png", "beta vs val total AUC")

    plot_comp_metric(tail_val_flux, "message_passing", "k_hop", "flux", OUT_DIR / "khop_tail_val_flux.png", "k-hop vs val flux (tail)")
    plot_comp_metric(tail_val_total, "message_passing", "k_hop", "total", OUT_DIR / "khop_tail_val_total.png", "k-hop vs val total (tail)")
    plot_comp_metric(auc_val_flux, "message_passing", "k_hop", "flux", OUT_DIR / "khop_auc_val_flux.png", "k-hop vs val flux AUC")
    plot_comp_metric(auc_val_total, "message_passing", "k_hop", "total", OUT_DIR / "khop_auc_val_total.png", "k-hop vs val total AUC")

    # Heatmaps
    plot_heat(losses, "loss", "lambda_flux", "flux", OUT_DIR / "heat_lambda_fluxloss.png", "lambda_flux val flux loss heatmap")
    plot_heat(losses, "loss", "lambda_flux", "total", OUT_DIR / "heat_lambda_totalloss.png", "lambda_flux val total loss heatmap")
    plot_heat(losses, "local_solver", "beta", "flux", OUT_DIR / "heat_beta_fluxloss.png", "beta val flux loss heatmap")
    plot_heat(losses, "message_passing", "k_hop", "flux", OUT_DIR / "heat_khop_fluxloss.png", "k-hop val flux loss heatmap")

    # Ablations
    plot_ablation(summary, "flux_layer", "use_flux", "test_l2_rel_mean", OUT_DIR / "ablation_useflux_testl2.png", "use_flux ablation (Test L2)")
    plot_ablation(summary, "flux_layer", "use_flux", "test_flux_residual_mean", OUT_DIR / "ablation_useflux_testflux.png", "use_flux ablation (Test Flux)")
    plot_ablation(summary, "train", "curriculum", "test_l2_rel_mean", OUT_DIR / "ablation_curriculum_testl2.png", "curriculum ablation (Test L2)")
    plot_ablation(summary, "train", "curriculum", "test_flux_residual_mean", OUT_DIR / "ablation_curriculum_testflux.png", "curriculum ablation (Test Flux)")

    # Generalization
    plot_ablation(summary, "generalization", "mesh", "test_l2_rel_mean", OUT_DIR / "gen_mesh_testl2.png", "mesh generalization (Test L2)")
    plot_ablation(summary, "generalization", "mesh", "test_flux_residual_mean", OUT_DIR / "gen_mesh_testflux.png", "mesh generalization (Test Flux)")

    # Stability
    plot_sensitivity(summary, "stability", "eval_noise", "test_flux_residual_mean", OUT_DIR / "stability_noise_vs_testflux.png", "noise vs Test Flux Residual")
    plot_sensitivity(summary, "stability", "eval_noise", "test_noisy_l2_rel_mean", OUT_DIR / "stability_noise_vs_testnoisyl2.png", "noise vs Test Noisy L2")
    if {"test_noisy_l2_rel_mean", "test_l2_rel_mean"}.issubset(summary.columns):
        s = summary[(summary["module"] == "stability") & summary["test_noisy_l2_rel_mean"].notna() & summary["test_l2_rel_mean"].notna()].copy()
        if not s.empty:
            s["ratio"] = s["test_noisy_l2_rel_mean"] / s["test_l2_rel_mean"]
            plot_sensitivity(s, "stability", "eval_noise", "ratio", OUT_DIR / "stability_noise_ratio.png", "noise vs (noisy/clean) L2 ratio")

    # Overview
    plot_pareto(summary, OUT_DIR / "overview_pareto.png")
    plot_overview_dot(summary, "test_l2_rel_mean", OUT_DIR / "overview_test_l2_dot.png", "Test L2 across configs")
    plot_overview_dot(summary, "test_flux_residual_mean", OUT_DIR / "overview_test_flux_dot.png", "Test Flux Residual across configs")
    plot_overview_dot(summary, "test_flux_residual_post_mean", OUT_DIR / "overview_test_flux_post_dot.png", "Test Flux Residual Post across configs")

    # Boxplots
    plot_box(summary, "test_l2_rel_mean", OUT_DIR / "box_module_testl2.png", "Module-wise Test L2")
    plot_box(summary, "test_flux_residual_mean", OUT_DIR / "box_module_testflux.png", "Module-wise Test Flux Residual")

    # Correlations: use tail-10 val losses
    corr_base = tail_val_flux.merge(summary, on=["module", "param", "value", "seed"], how="inner", suffixes=("_tail", "_metric"))
    plot_corr(corr_base, "flux", "test_flux_residual_mean", OUT_DIR / "corr_fluxloss_vs_testflux.png", "Val flux tail vs Test Flux Residual")
    plot_corr(corr_base, "flux", "test_l2_rel_mean", OUT_DIR / "corr_fluxloss_vs_testl2.png", "Val flux tail vs Test L2")
    corr_total = tail_val_total.merge(summary, on=["module", "param", "value", "seed"], how="inner")
    plot_corr(corr_total, "total", "test_l2_rel_mean", OUT_DIR / "corr_valtotal_vs_testl2.png", "Val total tail vs Test L2")

    # Delta plots
    plot_delta(summary, "loss", "lambda_flux", "test_l2_rel_mean", "1.0", OUT_DIR / "delta_lambda_testl2.png", "delta Test L2 vs lambda_flux", logx=True)
    plot_delta(summary, "loss", "lambda_flux", "test_flux_residual_mean", "1.0", OUT_DIR / "delta_lambda_testflux.png", "delta Test Flux vs lambda_flux", logx=True)
    plot_delta(summary, "local_solver", "beta", "test_l2_rel_mean", "0p5", OUT_DIR / "delta_beta_testl2.png", "delta Test L2 vs beta")
    plot_delta(summary, "message_passing", "k_hop", "test_l2_rel_mean", "4", OUT_DIR / "delta_khop_testl2.png", "delta Test L2 vs k-hop")

    # Top charts
    plot_top(summary, "test_l2_rel_mean", OUT_DIR / "top_test_l2.png", "Top Test L2 (lower is better)")
    plot_top(summary, "test_flux_residual_mean", OUT_DIR / "top_test_flux.png", "Top Test Flux Residual (lower is better)")

    # Heatmap across configs already generated earlier; keep if not present
    heatmap_path = OUT_DIR / "heatmap_metrics_across_configs.png"
    if not heatmap_path.exists():
        metrics_cols = ["test_l2_rel_mean", "test_flux_residual_mean", "test_flux_residual_post_mean"]
        metrics_cols = [c for c in metrics_cols if c in summary.columns]
        if metrics_cols:
            agg = summary.groupby(["module", "param", "value"])[metrics_cols].mean().reset_index()
            agg["config"] = agg.apply(lambda r: f"{r['module']}/{r['param']}={r['value']}", axis=1)
            heat = agg.set_index("config")[metrics_cols]
            heat_z = (heat - heat.mean()) / heat.std(ddof=0).replace(0, np.nan)
            heat_z = heat_z.fillna(0.0)
            plt.figure(figsize=(8, max(4, 0.2 * len(heat_z))))
            im = plt.imshow(heat_z.values, aspect="auto", cmap="coolwarm")
            plt.yticks(range(len(heat_z.index)), heat_z.index, fontsize=6)
            plt.xticks(range(len(heat_z.columns)), heat_z.columns, rotation=30, ha="right")
            plt.title("Metrics Across Configs (z-scored)")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            save_fig(heatmap_path)

    # Write missing report against PDF
    expected = []
    if PDF_PATH.exists():
        reader = PdfReader(str(PDF_PATH))
        all_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        expected = sorted(set(re.findall(r"\\b[\\w\\-]+\\.png\\b", all_text)))

    produced = sorted([p.name for p in OUT_DIR.glob("*.png")])
    missing = [f for f in expected if f not in produced]
    extra = [f for f in produced if f not in expected]
    with open(OUT_DIR / "missing_vs_pdf.txt", "w", encoding="utf-8") as f:
        f.write("Expected PNGs from PDF:\n")
        f.write("\n".join(expected))
        f.write("\n\nProduced PNGs:\n")
        f.write("\n".join(produced))
        f.write("\n\nMissing vs PDF:\n")
        f.write("\n".join(missing))
        f.write("\n\nExtra (not in PDF):\n")
        f.write("\n".join(extra))


if __name__ == "__main__":
    main()
