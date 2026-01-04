#!/usr/bin/env python3
"""
Rerun missing experiments listed in missing_runs.csv for a given run root.
"""

from __future__ import annotations

import csv
import os
import subprocess
import time
from typing import Dict, Tuple, Any

import yaml
from pathlib import Path


def _slug_to_value(value: str) -> Any:
    if value == "on":
        return True
    if value == "off":
        return False
    if value.isdigit():
        return int(value)
    if value.count("p") == 1 and value.replace("p", "").isdigit():
        left, right = value.split("p", 1)
        return float(f"{left}.{right}")
    return value


def _overrides_map() -> Dict[Tuple[str, str], Dict[Any, Dict[str, object]]]:
    return {
        ("flux_layer", "use_flux"): {
            True: {"disable_flux": False},
            False: {"disable_flux": True, "lambda_flux": 0.0},
        },
        ("loss", "lambda_flux"): {
            0.0: {"lambda_flux": 0.0, "disable_flux": False},
            1e-4: {"lambda_flux": 1e-4, "disable_flux": False},
            3e-4: {"lambda_flux": 3e-4, "disable_flux": False},
            1e-3: {"lambda_flux": 1e-3, "disable_flux": False},
            3e-3: {"lambda_flux": 3e-3, "disable_flux": False},
            1e-2: {"lambda_flux": 1e-2, "disable_flux": False},
            3e-2: {"lambda_flux": 3e-2, "disable_flux": False},
            0.1: {"lambda_flux": 0.1, "disable_flux": False},
            0.3: {"lambda_flux": 0.3, "disable_flux": False},
            1.0: {"lambda_flux": 1.0, "disable_flux": False},
        },
        ("local_solver", "beta"): {
            0.0: {"local_beta": 0.0},
            0.05: {"local_beta": 0.05},
            0.1: {"local_beta": 0.1},
            0.2: {"local_beta": 0.2},
            0.35: {"local_beta": 0.35},
            0.5: {"local_beta": 0.5},
            0.65: {"local_beta": 0.65},
        },
        ("message_passing", "k_hop"): {
            1: {"edge_csv": "dataset/edges_knn_k1.csv"},
            2: {"edge_csv": "dataset/edges_knn_k2.csv"},
            3: {"edge_csv": "dataset/edges_knn_k3.csv"},
            4: {"edge_csv": "dataset/edges_knn_k4.csv"},
            6: {"edge_csv": "dataset/edges_knn_k6.csv"},
            8: {"edge_csv": "dataset/edges_knn_k8.csv", "batch_size": 1},
        },
        ("graph_ode", "nstep"): {
            4: {"ode_steps": 4},
            8: {"ode_steps": 8},
            16: {"ode_steps": 16},
        },
        ("graph_ode", "solver"): {
            "rk4": {"ode_method": "rk4"},
            "dopri5": {"ode_method": "dopri5", "ode_rtol": 1e-3, "ode_atol": 1e-6},
        },
        ("flux_layer", "nproj"): {
            1: {"flux_nproj": 1},
            3: {"flux_nproj": 3},
            5: {"flux_nproj": 5},
        },
        ("train", "curriculum"): {
            "pretrain": {"pretrain_epochs": 10},
            "joint": {"pretrain_epochs": 0},
        },
        ("generalization", "mesh"): {
            "train_k4_test_k8": {
                "train_edge_csv": "dataset/edges_knn_k4.csv",
                "val_edge_csv": "dataset/edges_knn_k4.csv",
                "test_edge_csv": "dataset/edges_knn_k8.csv",
                "batch_size": 1,
            },
            "train_k8_test_k4": {
                "train_edge_csv": "dataset/edges_knn_k8.csv",
                "val_edge_csv": "dataset/edges_knn_k8.csv",
                "test_edge_csv": "dataset/edges_knn_k4.csv",
                "batch_size": 1,
            },
        },
        ("stability", "eval_noise"): {
            0.0: {"eval_noise_std": 0.0},
            0.005: {"eval_noise_std": 0.005},
            0.01: {"eval_noise_std": 0.01},
        },
        ("aggressive", "config"): {
            "A": {
                "hidden_dim": 128,
                "num_regions": 4,
                "local_depth": 3,
                "local_beta": 0.8,
                "ode_steps": 8,
                "ode_t": 2.0,
                "flux_nproj": 3,
                "lambda_flux": 5.0,
                "batch_size": 1,
                "lr": 3e-4,
            },
            "B": {
                "hidden_dim": 64,
                "num_regions": 4,
                "local_depth": 3,
                "local_beta": 0.8,
                "ode_steps": 8,
                "ode_t": 2.0,
                "flux_nproj": 3,
                "lambda_flux": 5.0,
                "edge_csv": "dataset/edges_knn_k8.csv",
                "batch_size": 1,
                "lr": 3e-4,
            },
            "C": {
                "hidden_dim": 64,
                "num_regions": 4,
                "local_depth": 3,
                "local_beta": 0.8,
                "ode_method": "dopri5",
                "ode_t": 2.0,
                "ode_rtol": 1e-4,
                "ode_atol": 1e-6,
                "ode_steps": 4,
                "flux_nproj": 5,
                "lambda_flux": 5.0,
                "batch_size": 1,
                "lr": 3e-4,
            },
        },
    }


def _ensure_config(repo_root: Path, run_root: Path, row: Dict[str, str]) -> Path:
    module = row["module"]
    param = row["param"]
    value_slug = row["value"]
    seed = int(row["seed"])
    cfg_path = run_root / module / param / value_slug / f"seed{seed}" / "config.yaml"
    if cfg_path.exists():
        return cfg_path

    base_cfg = yaml.safe_load((repo_root / "baseline.yaml").read_text(encoding="utf-8"))
    overrides = _overrides_map().get((module, param), {})
    value = _slug_to_value(value_slug)
    cfg = dict(base_cfg)
    cfg.update(overrides.get(value, {}))
    cfg["seed"] = seed
    out_dir = cfg_path.parent
    cfg["output_dir"] = str(out_dir)
    cfg["loss_csv"] = str(out_dir / "losses.csv")
    cfg["ckpt"] = str(out_dir / "model.pt")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return cfg_path


def main() -> None:
    run_root_env = os.getenv("RUN_ROOT")
    if not run_root_env:
        raise SystemExit("RUN_ROOT is required (e.g. /root/autodl-tmp/CreativeGNN/exp_runs/...)")
    run_root = Path(run_root_env)
    repo_root = Path(__file__).resolve().parents[1]
    missing_csv = run_root / "missing_runs.csv"
    if not missing_csv.exists():
        raise SystemExit(f"Missing {missing_csv}")

    with missing_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No missing runs to execute.")
        return

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    max_retries = int(env.get("MAX_RETRIES", "0"))
    failed = []

    for row in rows:
        module = row["module"]
        param = row["param"]
        value = row["value"]
        seed = row["seed"]
        cfg_path = _ensure_config(repo_root, run_root, row)
        metrics_path = cfg_path.parent / "metrics.json"
        if metrics_path.exists():
            print(f"Already done: {metrics_path}")
            continue
        print(f"Running {cfg_path}")
        ok = False
        for attempt in range(max_retries + 1):
            proc = subprocess.run(
                ["python", "starttraining.py", "-c", str(cfg_path)],
                env=env,
            )
            if proc.returncode == 0:
                ok = True
                break
            time.sleep(2)
        if not ok:
            failed.append(row)
            print(f"Failed: {cfg_path}")

    if failed:
        failed_path = run_root / "failed_runs.csv"
        with failed_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(failed)
        print(f"Wrote failed runs to {failed_path}")


if __name__ == "__main__":
    main()

