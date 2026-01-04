#!/usr/bin/env python3
"""
Aggregate the experiment data under exp/ and create comparison charts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_SUBPATH = Path("exp")


def experiment_name(file_path: Path, exp_root: Path) -> str:
    rel = file_path.relative_to(exp_root)
    config_parts = rel.parts[:-2]
    if config_parts:
        return "/".join(config_parts)
    return rel.parts[0]


def load_metrics(exp_root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(exp_root.rglob("metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Skipping {path} due to {exc!r}")
            continue
        experiment = experiment_name(path, exp_root)
        for split in ("val", "test"):
            metrics = payload.get(split)
            if not metrics:
                continue
            record = {
                "experiment": experiment,
                "seed": payload.get("seed"),
                "split": split,
            }
            record.update(metrics)
            records.append(record)
    return pd.DataFrame(records)


def load_final_losses(exp_root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(exp_root.rglob("losses.csv")):
        df = pd.read_csv(path)
        experiment = experiment_name(path, exp_root)
        df_finetune = df[df["stage"] == "finetune"]
        if df_finetune.empty:
            continue
        for split in ("train", "val"):
            subset = df_finetune[df_finetune["split"] == split]
            if subset.empty:
                continue
            row = subset.iloc[-1]
            record = {
                "experiment": experiment,
                "seed": path.parent.name,
                "split": split,
            }
            for col in ("supervised", "flux", "vorticity", "smooth", "total"):
                record[col] = float(row[col])
            records.append(record)
    return pd.DataFrame(records)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    summary = (
        metrics_df.groupby(["experiment", "split"])
        .agg(
            l2_rel_mean_mean=("l2_rel_mean", "mean"),
            l2_rel_mean_std=("l2_rel_mean", "std"),
            flux_residual_mean_mean=("flux_residual_mean", "mean"),
            flux_residual_mean_std=("flux_residual_mean", "std"),
            time_infer_s_mean=("time_infer_s", "mean"),
        )
        .reset_index()
    )
    summary["l2_rel_mean_std"] = summary["l2_rel_mean_std"].fillna(0.0)
    summary["flux_residual_mean_std"] = summary["flux_residual_mean_std"].fillna(0.0)
    return summary


def summarize_losses(loss_df: pd.DataFrame) -> pd.DataFrame:
    if loss_df.empty:
        return loss_df
    summary = (
        loss_df.groupby(["experiment", "split"])
        .agg(
            supervised_mean=("supervised", "mean"),
            flux_mean=("flux", "mean"),
            total_mean=("total", "mean"),
            total_std=("total", "std"),
        )
        .reset_index()
    )
    summary["total_std"] = summary["total_std"].fillna(0.0)
    return summary


def best_worst_test(metrics_df: pd.DataFrame) -> Iterable[str]:
    lines = []
    test_rows = metrics_df[metrics_df["split"] == "test"]
    if test_rows.empty:
        return lines
    best = test_rows.loc[test_rows["l2_rel_mean"].idxmin()]
    worst = test_rows.loc[test_rows["l2_rel_mean"].idxmax()]
    lines.append(
        f"Best test l2_rel_mean {best['l2_rel_mean']:.3f} "
        f"({best['experiment']} seed {int(best['seed'])})"
    )
    lines.append(
        f"Worst test l2_rel_mean {worst['l2_rel_mean']:.3f} "
        f"({worst['experiment']} seed {int(worst['seed'])})"
    )
    return lines


def draw_charts(
    metrics_summary: pd.DataFrame,
    loss_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    test_summary = metrics_summary[metrics_summary["split"] == "test"]
    experiments = sorted(test_summary["experiment"].unique())
    if not experiments:
        return

    positions = np.arange(len(experiments))
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    l2_means = test_summary.set_index("experiment").reindex(experiments)["l2_rel_mean_mean"]
    l2_err = (
        test_summary.set_index("experiment").reindex(experiments)["l2_rel_mean_std"].fillna(0.0)
    )
    axes[0].bar(positions, l2_means, color="#1f77b4")
    axes[0].errorbar(positions, l2_means, yerr=l2_err, fmt="none", color="black", capsize=5)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(experiments, rotation=45, ha="right")
    axes[0].set_ylabel("Test l2_rel_mean")
    axes[0].set_title("Test Relative L2 Error by Experiment")

    flux_means = (
        test_summary.set_index("experiment").reindex(experiments)["flux_residual_mean_mean"]
    )
    flux_err = (
        test_summary.set_index("experiment").reindex(experiments)["flux_residual_mean_std"].fillna(0.0)
    )
    axes[1].bar(positions, flux_means, color="#ff7f0e")
    axes[1].errorbar(
        positions, flux_means, yerr=flux_err, fmt="none", color="black", capsize=5
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(experiments, rotation=45, ha="right")
    axes[1].set_ylabel("Test Flux Residual")
    axes[1].set_title("Test Flux Residual Mean by Experiment")

    loss_plot = loss_summary[loss_summary["experiment"].isin(experiments)]
    if loss_plot.empty:
        axes[2].text(0.5, 0.5, "No finetune loss logs available", ha="center", va="center")
        axes[2].set_xticks(positions)
        axes[2].set_xticklabels(experiments, rotation=45, ha="right")
        axes[2].set_ylabel("Finetune Total Loss (last epoch)")
        axes[2].set_title("Final Finetune Loss by Experiment")
    else:
        loss_pivot = loss_plot.pivot(index="experiment", columns="split")
        total_mean = loss_pivot["total_mean"].reindex(index=experiments).reindex(
            columns=["train", "val"], fill_value=np.nan
        )
        total_std = loss_pivot["total_std"].reindex(index=experiments).reindex(
            columns=["train", "val"], fill_value=0.0
        )

        width = 0.35
        axes[2].bar(
            positions - width / 2,
            total_mean["train"],
            width,
            label="train",
            color="#2ca02c",
        )
        axes[2].bar(
            positions + width / 2,
            total_mean["val"],
            width,
            label="val",
            color="#d62728",
        )
        axes[2].errorbar(
            positions - width / 2,
            total_mean["train"],
            yerr=total_std["train"],
            fmt="none",
            color="black",
            capsize=5,
        )
        axes[2].errorbar(
            positions + width / 2,
            total_mean["val"],
            yerr=total_std["val"],
            fmt="none",
            color="black",
            capsize=5,
        )
        axes[2].set_xticks(positions)
        axes[2].set_xticklabels(experiments, rotation=45, ha="right")
        axes[2].set_ylabel("Finetune Total Loss (last epoch)")
        axes[2].set_title("Final Finetune Loss by Experiment")
        axes[2].legend()

    plt.tight_layout()
    chart_path = output_dir / "exp_metrics_comparison.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    print(f"Saved comparison chart to {chart_path}")


def slug_to_float(slug: str) -> float | None:
    if not slug:
        return None
    if slug.count("p") == 1 and slug.replace("p", "").isdigit():
        left, right = slug.split("p", 1)
        return float(f"{left}.{right}")
    try:
        return float(slug)
    except ValueError:
        return None


def load_lambda_flux_losses(exp_root: Path) -> pd.DataFrame:
    records = []
    for path in sorted((exp_root / "loss").rglob("losses.csv")):
        parts = path.parts
        if "lambda_flux" not in parts:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover
            print(f"Skipping {path} due to {exc!r}")
            continue
        slug = path.parent.parent.name
        lambda_flux = slug_to_float(slug)
        if lambda_flux is None:
            cfg_path = path.parent / "config.yaml"
            if cfg_path.exists():
                for line in cfg_path.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("lambda_flux:"):
                        try:
                            lambda_flux = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            lambda_flux = None
                        break
        for _, row in df.iterrows():
            records.append(
                {
                    "lambda_flux": lambda_flux,
                    "seed": path.parent.name,
                    "epoch": int(row["epoch"]),
                    "stage": row["stage"],
                    "split": row["split"],
                    "flux": float(row["flux"]),
                }
            )
    return pd.DataFrame(records)


def draw_lambda_flux_sensitivity(loss_df: pd.DataFrame, output_dir: Path) -> None:
    if loss_df.empty:
        print("No lambda_flux loss records found for sensitivity plot.")
        return

    df = loss_df.dropna(subset=["lambda_flux"])
    if df.empty:
        print("No lambda_flux values resolved for sensitivity plot.")
        return

    df = df[df["stage"] == "finetune"]
    if df.empty:
        print("No finetune rows found for sensitivity plot.")
        return

    df = df[df["split"].isin(["train", "val"])]
    if df.empty:
        print("No train/val rows found for sensitivity plot.")
        return

    output_csv = output_dir / "lambda_flux_flux_loss_points.csv"
    df.to_csv(output_csv, index=False)
    print(f"Wrote lambda_flux flux-loss points to {output_csv}")

    lambda_vals = sorted(df["lambda_flux"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    splits = ["train", "val"]
    colors = {"train": "#1f77b4", "val": "#ff7f0e"}

    for ax, split in zip(axes, splits):
        split_df = df[df["split"] == split]
        if split_df.empty:
            ax.set_title(f"{split} (no data)")
            continue
        for idx, val in enumerate(lambda_vals):
            vals = split_df[split_df["lambda_flux"] == val]["flux"].values
            if vals.size == 0:
                continue
            jitter = (np.random.rand(vals.size) - 0.5) * 0.06
            ax.scatter(
                np.full(vals.size, val) + jitter,
                vals,
                s=12,
                alpha=0.6,
                color=colors[split],
                edgecolors="none",
            )
        means = split_df.groupby("lambda_flux")["flux"].mean().reindex(lambda_vals)
        stds = split_df.groupby("lambda_flux")["flux"].std().reindex(lambda_vals).fillna(0.0)
        ax.errorbar(lambda_vals, means, yerr=stds, fmt="-o", color="black", capsize=4)
        ax.set_xlabel("λ_flux (physics loss weight)")
        ax.set_title(f"Sensitivity: λ_flux vs {split} flux loss (finetune)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Flux Loss")
    plt.tight_layout()
    chart_path = output_dir / "lambda_flux_sensitivity_flux_loss.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    print(f"Saved lambda_flux sensitivity chart to {chart_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exp_root = repo_root / EXP_SUBPATH
    if not exp_root.exists():
        raise SystemExit(f"Expected exp directory at {exp_root}")

    output_dir = repo_root / "exp_reports"
    output_dir.mkdir(exist_ok=True)

    metrics_df = load_metrics(exp_root)
    loss_df = load_final_losses(exp_root)

    metrics_summary = summarize_metrics(metrics_df)
    loss_summary = summarize_losses(loss_df)

    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    loss_summary.to_csv(output_dir / "finetune_loss_summary.csv", index=False)
    print(f"Wrote metrics summary to {output_dir / 'metrics_summary.csv'}")
    print(f"Wrote finetune loss summary to {output_dir / 'finetune_loss_summary.csv'}")

    draw_charts(metrics_summary, loss_summary, output_dir)

    lambda_flux_losses = load_lambda_flux_losses(exp_root)
    draw_lambda_flux_sensitivity(lambda_flux_losses, output_dir)

    for line in best_worst_test(metrics_df):
        print(line)


if __name__ == "__main__":
    main()
