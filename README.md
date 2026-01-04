Creative-GNN: Physics-Embedded Graph Neural ODE for Aerodynamic Flow Fields
============================================================================

This repository implements the Creative-GNN architecture described in the paper. It provides a PyTorch-based reference with continuous-depth graph dynamics, region-adaptive local solvers, and an explicit flux conservation layer, together with physics-informed losses for mass, momentum/energy proxy, and vorticity.

## Quick Start
- Install dependencies: `pip install -r requirements.txt`
- Run a smoke test with synthetic graphs: `python train.py --epochs 2`
- For config-driven training, edit `config.yaml` then run `python starttraining.py -c config.yaml`.
- Replace the synthetic dataset with your CFD-derived graphs (see `src/data.py` for expected fields).

## Project Layout
- `src/models/creative_gnn.py`: end-to-end model (encode → ODE evolve → local solver → flux correction → decode).
- `src/models/graph_ode.py`: continuous-time message passing (Graph Neural ODE).
- `src/models/local_solver.py`: region-aware local solvers for shocks/boundary layers/wakes.
- `src/models/flux_layer.py`: edge flux prediction and node-wise conservation residuals.
- `src/models/mlp.py`: small shared MLP utilities.
- `src/losses.py`: supervised + physics losses (flux, vorticity, smoothness).
- `src/data.py`: graph data container, synthetic dataset, and CSV-based loading template.
- `train.py`: training/evaluation loop with logging and checkpointing hooks.

## Using Real CFD Data
Default format is aligned to the schema below (sample CSVs are not committed):
- Node CSV columns (sanitized names): `Node_Number, X__m_, Y__m_, Z__m_, Density__kg_m3_, Eddy_Viscosity__Pa_s_, Pressure__Pa_, Static_Enthalpy__J_kg1_, Temperature__K_, Velocity_u__m_s1_, Velocity_v__m_s1_, Velocity_w__m_s1_, VelocityCurl_X__s1_, VelocityCurl_Y__s1_, VelocityCurl_Z__s1_`
- Edge CSV columns: `row_index, col_index, value` (value ignored), indices are assumed 0-based.

Run training on these CSVs (defaults predict pressure + u/v/w + density from density/eddy_viscosity/pressure/enthalpy/temperature):
```bash
python train.py --node-csv "dataset/nodes.csv" --edge-csv "dataset/edges.csv" --epochs 2 --batch-size 1 --device cpu
```
Adjust input/target columns if needed:
```bash
python train.py --node-csv "dataset/nodes.csv" --edge-csv "dataset/edges.csv" ^
  --in-fields Density__kg_m3_,Eddy_Viscosity__Pa_s_,Pressure__Pa_,Static_Enthalpy__J_kg1_,Temperature__K_ ^
  --out-fields Pressure__Pa_,Velocity_u__m_s1_,Velocity_v__m_s1_,Velocity_w__m_s1_,Density__kg_m3_
```
If you have multiple cases, pass lists in `CFDDataset` or create multiple runs.

### Using dataset/ folder (file names encode Mach and tan(attack angle))
Node CSVs can be stored under `dataset/` (gitignored) with names like `Ma=2 t=0.csv`, `Ma=2 t=0.1.csv`, `Ma=2.5 t=0.2.csv` (Ma = Mach, t = tan(angle of attack)). Edges can be shared (e.g., `dataset/edges.csv`). You can glob multiple cases at once:
```bash
python train.py --node-csv "dataset/Ma=2*.csv" --edge-csv "dataset/edges.csv" --batch-size 1 --device cpu
```
If you have one edge file and multiple node files, the edge path will be reused automatically. Comma-separated lists also work: `--node-csv "dataset/Ma=2 t=0.csv,dataset/Ma=2 t=0.1.csv"`.

### Config-based quick start
Edit `config.yaml` to set data paths and hyperparameters, then run:
```bash
python starttraining.py -c config.yaml
```
For fixed train/val/test splits, see `baseline.yaml` and run:
```bash
python starttraining.py -c baseline.yaml
```

### If you already have a CGNS mesh, reindex the node IDs in your CSV. 
Fluent exports can reorder nodes, so align the CSV with the mesh file:
```bash
python reorder_by_cgns.py --cgns mesh.cgns --node-csv "dataset/Ma=2 t=0.csv" --out-csv "dataset/Ma=2 t=0_reordered.csv" \
  --edge-csv "dataset/edges.csv" --out-edge-csv "dataset/edges_reordered.csv" --tol 1e-6
```
The script reorders the node CSV to match the CGNS node order; if an edge file is provided, it remaps indices as well. Make sure coordinates match (tol is the matching tolerance).

## Notes
- The reference integration uses `torchdiffeq` (`odeint`) with RK4 by default; swap to an adaptive solver for stiff regimes.
- Flux and vorticity losses use geometric edge differences; ensure coordinates are in consistent units and appropriately scaled.
- Region masks can be heuristically built from CFD fields (e.g., density gradient for shocks, wall distance for boundary layers).
- To record losses per epoch, pass `--loss-csv path.csv` or set `loss_csv` in `config.yaml`.
- When `output_dir` is set, `config.yaml`, `train.log`, and `metrics.json` are written there.
