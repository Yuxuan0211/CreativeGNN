import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml


def _slug(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value)


def _write_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True)


def _run(cfg_path: Path, env: Dict[str, str] | None = None) -> None:
    subprocess.run(["python", "starttraining.py", "-c", str(cfg_path)], check=True, env=env)


def main() -> None:
    base_cfg = yaml.safe_load(Path("baseline.yaml").read_text(encoding="utf-8"))
    seeds = [0, 1, 2]
    force_rerun = os.getenv("FORCE_RERUN", "1") == "1"
    run_root_env = os.getenv("RUN_ROOT")
    if run_root_env:
        run_root = Path(run_root_env)
    else:
        run_root = Path("exp_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    experiments: List[Dict] = []

    # 1) Flux layer on/off
    experiments.append(
        {
            "module": "flux_layer",
            "param": "use_flux",
            "values": [True, False],
            "overrides": {
                True: {"disable_flux": False},
                False: {"disable_flux": True, "lambda_flux": 0.0},
            },
        }
    )

    # 2) lambda_flux sweep
    experiments.append(
        {
            "module": "loss",
            "param": "lambda_flux",
            "values": [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 3e-1, 1.0],
            "overrides": {
                0.0: {"lambda_flux": 0.0, "disable_flux": False},
                1e-4: {"lambda_flux": 1e-4, "disable_flux": False},
                3e-4: {"lambda_flux": 3e-4, "disable_flux": False},
                1e-3: {"lambda_flux": 1e-3, "disable_flux": False},
                3e-3: {"lambda_flux": 3e-3, "disable_flux": False},
                1e-2: {"lambda_flux": 1e-2, "disable_flux": False},
                3e-2: {"lambda_flux": 3e-2, "disable_flux": False},
                0.1: {"lambda_flux": 0.1, "disable_flux": False},
                3e-1: {"lambda_flux": 3e-1, "disable_flux": False},
                1.0: {"lambda_flux": 1.0, "disable_flux": False},
            },
        }
    )

    # 3) Local solver beta
    experiments.append(
        {
            "module": "local_solver",
            "param": "beta",
            "values": [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65],
            "overrides": {
                0.0: {"local_beta": 0.0},
                0.05: {"local_beta": 0.05},
                0.1: {"local_beta": 0.1},
                0.2: {"local_beta": 0.2},
                0.35: {"local_beta": 0.35},
                0.5: {"local_beta": 0.5},
                0.65: {"local_beta": 0.65},
            },
        }
    )

    # 4) k-hop (proxy via kNN edges)
    experiments.append(
        {
            "module": "message_passing",
            "param": "k_hop",
            "values": [1, 2, 3, 4, 6, 8],
            "overrides": {
                1: {"edge_csv": "dataset/edges_knn_k1.csv"},
                2: {"edge_csv": "dataset/edges_knn_k2.csv"},
                3: {"edge_csv": "dataset/edges_knn_k3.csv"},
                4: {"edge_csv": "dataset/edges_knn_k4.csv"},
                6: {"edge_csv": "dataset/edges_knn_k6.csv"},
                8: {"edge_csv": "dataset/edges_knn_k8.csv", "batch_size": 1},
            },
        }
    )

    # 5) ODE steps
    experiments.append(
        {
            "module": "graph_ode",
            "param": "nstep",
            "values": [4, 8, 16],
            "overrides": {
                4: {"ode_steps": 4},
                8: {"ode_steps": 8},
                16: {"ode_steps": 16},
            },
        }
    )

    # 6) ODE solver method
    experiments.append(
        {
            "module": "graph_ode",
            "param": "solver",
            "values": ["rk4", "dopri5"],
            "overrides": {
                "rk4": {"ode_method": "rk4"},
                "dopri5": {"ode_method": "dopri5", "ode_rtol": 1e-3, "ode_atol": 1e-6},
            },
        }
    )

    # 7) Flux projection iterations
    experiments.append(
        {
            "module": "flux_layer",
            "param": "nproj",
            "values": [1, 3, 5],
            "overrides": {
                1: {"flux_nproj": 1},
                3: {"flux_nproj": 3},
                5: {"flux_nproj": 5},
            },
        }
    )

    # 8) Curriculum vs joint
    experiments.append(
        {
            "module": "train",
            "param": "curriculum",
            "values": ["pretrain", "joint"],
            "overrides": {
                "pretrain": {"pretrain_epochs": base_cfg.get("pretrain_epochs", 10)},
                "joint": {"pretrain_epochs": 0},
            },
        }
    )

    # 9) Mesh generalization: train vs test edges
    experiments.append(
        {
            "module": "generalization",
            "param": "mesh",
            "values": ["train_k4_test_k8", "train_k8_test_k4"],
            "overrides": {
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
        }
    )

    # 10) Perturbation robustness (eval noise)
    experiments.append(
        {
            "module": "stability",
            "param": "eval_noise",
            "values": [0.0, 0.005, 0.01],
            "overrides": {
                0.0: {"eval_noise_std": 0.0},
                0.005: {"eval_noise_std": 0.005},
                0.01: {"eval_noise_std": 0.01},
            },
        }
    )

    # 11) Aggressive configs (capacity + physics + horizon)
    experiments.append(
        {
            "module": "aggressive",
            "param": "config",
            "values": ["A", "B", "C"],
            "overrides": {
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
    )

    summary_rows: List[Dict[str, object]] = []
    summary_path = run_root / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Include existing baseline runs if present
    for seed in seeds:
        metrics_path = run_root / "baseline" / f"seed{seed}" / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary_rows.append(
                {
                    "module": "baseline",
                    "param": "baseline",
                    "value": "baseline",
                    "seed": seed,
                    "val_l2_rel_mean": metrics.get("val", {}).get("l2_rel_mean"),
                    "val_flux_residual_mean": metrics.get("val", {}).get("flux_residual_mean"),
                    "val_flux_residual_post_mean": metrics.get("val", {}).get("flux_residual_post_mean"),
                    "test_l2_rel_mean": metrics.get("test", {}).get("l2_rel_mean"),
                    "test_flux_residual_mean": metrics.get("test", {}).get("flux_residual_mean"),
                    "test_flux_residual_post_mean": metrics.get("test", {}).get("flux_residual_post_mean"),
                    "eval_noise_std": metrics.get("eval_noise_std", 0.0),
                    "val_noisy_l2_rel_mean": metrics.get("val_noisy", {}).get("l2_rel_mean"),
                    "test_noisy_l2_rel_mean": metrics.get("test_noisy", {}).get("l2_rel_mean"),
                }
            )

    for exp in experiments:
        module = exp["module"]
        param = exp["param"]
        for value in exp["values"]:
            overrides = exp["overrides"][value]
            for seed in seeds:
                cfg = dict(base_cfg)
                cfg.update(overrides)
                cfg["seed"] = seed
                value_slug = _slug(value)
                out_dir = run_root / module / param / value_slug / f"seed{seed}"
                cfg["output_dir"] = str(out_dir)
                cfg["loss_csv"] = str(out_dir / "losses.csv")
                cfg["ckpt"] = str(out_dir / "model.pt")
                cfg_path = out_dir / "config.yaml"
                _write_yaml(cfg_path, cfg)

                metrics_path = out_dir / "metrics.json"
                if force_rerun and metrics_path.exists():
                    metrics_path.unlink()
                if not metrics_path.exists():
                    env = os.environ.copy()
                    try:
                        _run(cfg_path, env=env)
                    except subprocess.CalledProcessError:
                        # Retry once with smaller batch size and allocator hint
                        if cfg.get("batch_size", 2) > 1:
                            cfg["batch_size"] = 1
                            _write_yaml(cfg_path, cfg)
                        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                        _run(cfg_path, env=env)

                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    summary_rows.append(
                        {
                            "module": module,
                            "param": param,
                            "value": value,
                            "seed": seed,
                            "val_l2_rel_mean": metrics.get("val", {}).get("l2_rel_mean"),
                            "val_flux_residual_mean": metrics.get("val", {}).get("flux_residual_mean"),
                            "val_flux_residual_post_mean": metrics.get("val", {}).get("flux_residual_post_mean"),
                            "test_l2_rel_mean": metrics.get("test", {}).get("l2_rel_mean"),
                            "test_flux_residual_mean": metrics.get("test", {}).get("flux_residual_mean"),
                            "test_flux_residual_post_mean": metrics.get("test", {}).get("flux_residual_post_mean"),
                            "eval_noise_std": metrics.get("eval_noise_std", 0.0),
                            "val_noisy_l2_rel_mean": metrics.get("val_noisy", {}).get("l2_rel_mean"),
                            "test_noisy_l2_rel_mean": metrics.get("test_noisy", {}).get("l2_rel_mean"),
                        }
                    )

    if summary_rows:
        headers = list(summary_rows[0].keys())
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row[h]) for h in headers) + "\n")

        # Aggregate mean/std by (module, param, value)
        agg: Dict[str, Dict[str, List[float]]] = {}
        for row in summary_rows:
            key = f"{row['module']}|{row['param']}|{row['value']}"
            agg.setdefault(key, {})
            for metric in [
                "val_l2_rel_mean",
                "test_l2_rel_mean",
                "val_flux_residual_mean",
                "test_flux_residual_mean",
                "val_noisy_l2_rel_mean",
                "test_noisy_l2_rel_mean",
            ]:
                val = row.get(metric)
                if val is None or val == "None":
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                agg[key].setdefault(metric, []).append(fval)

        agg_path = run_root / "summary_agg.csv"
        with agg_path.open("w", encoding="utf-8") as f:
            f.write(
                "module,param,value,"
                "val_l2_rel_mean_mean,val_l2_rel_mean_std,"
                "test_l2_rel_mean_mean,test_l2_rel_mean_std,"
                "val_flux_residual_mean_mean,val_flux_residual_mean_std,"
                "test_flux_residual_mean_mean,test_flux_residual_mean_std,"
                "val_noisy_l2_rel_mean_mean,val_noisy_l2_rel_mean_std,"
                "test_noisy_l2_rel_mean_mean,test_noisy_l2_rel_mean_std\n"
            )
            for key, metrics in sorted(agg.items()):
                module, param, value = key.split("|", 2)

                def _mean_std(vals: List[float]) -> List[str]:
                    if not vals:
                        return ["None", "None"]
                    mean = sum(vals) / len(vals)
                    var = sum((v - mean) ** 2 for v in vals) / len(vals)
                    std = var ** 0.5
                    return [str(mean), str(std)]

                row = [
                    module,
                    param,
                    value,
                    *_mean_std(metrics.get("val_l2_rel_mean", [])),
                    *_mean_std(metrics.get("test_l2_rel_mean", [])),
                    *_mean_std(metrics.get("val_flux_residual_mean", [])),
                    *_mean_std(metrics.get("test_flux_residual_mean", [])),
                    *_mean_std(metrics.get("val_noisy_l2_rel_mean", [])),
                    *_mean_std(metrics.get("test_noisy_l2_rel_mean", [])),
                ]
                f.write(",".join(row) + "\n")


if __name__ == "__main__":
    main()

