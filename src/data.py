import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class GraphSample:
    """Single graph container."""
    x: torch.Tensor  # (N, F_in)
    pos: torch.Tensor  # (N, 3)
    edge_index: torch.Tensor  # (2, E) long
    y: torch.Tensor  # (N, F_out)
    region_mask: Optional[torch.Tensor] = None  # (N, R) soft weights
    volume: Optional[torch.Tensor] = None  # (N,) optional control-volume weights


@dataclass
class GraphBatch(GraphSample):
    batch: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))  # (N,) graph index
    num_graphs: int = 1


def collate_graphs(samples: Sequence[GraphSample]) -> GraphBatch:
    """Concatenate graphs for mini-batch training."""
    xs, poss, ys, masks, vols, edge_indices, batch = [], [], [], [], [], [], []
    node_offset = 0
    for graph_id, g in enumerate(samples):
        n = g.x.size(0)
        xs.append(g.x)
        poss.append(g.pos)
        ys.append(g.y)
        masks.append(g.region_mask if g.region_mask is not None else None)
        vols.append(g.volume if g.volume is not None else None)
        edge_indices.append(g.edge_index + node_offset)
        batch.append(torch.full((n,), graph_id, dtype=torch.long))
        node_offset += n

    x = torch.cat(xs, dim=0)
    pos = torch.cat(poss, dim=0)
    y = torch.cat(ys, dim=0)
    edge_index = torch.cat(edge_indices, dim=1)
    batch_idx = torch.cat(batch, dim=0)

    # Region masks: pad missing with uniform weights
    if any(m is None for m in masks):
        max_r = max((m.shape[1] if m is not None else 0) for m in masks)
        padded_masks = []
        for m, n in zip(masks, [g.x.size(0) for g in samples]):
            if m is None or m.shape[1] != max_r:
                padded_masks.append(torch.full((n, max_r), 1.0 / max_r if max_r > 0 else 0.0))
            else:
                padded_masks.append(m)
        region_mask = torch.cat(padded_masks, dim=0) if max_r > 0 else None
    else:
        region_mask = torch.cat(masks, dim=0) if masks else None

    if any(v is not None for v in vols):
        padded_vols = []
        for v, n in zip(vols, [g.x.size(0) for g in samples]):
            if v is None:
                padded_vols.append(torch.ones(n))
            else:
                padded_vols.append(v)
        volume = torch.cat(padded_vols, dim=0)
    else:
        volume = None

    return GraphBatch(
        x=x,
        pos=pos,
        edge_index=edge_index,
        y=y,
        region_mask=region_mask,
        volume=volume,
        batch=batch_idx,
        num_graphs=len(samples),
    )


class SyntheticFlowDataset(Dataset):
    """Generates toy graphs mimicking shock/boundary-layer regions."""

    def __init__(self, num_graphs: int = 64, seed: int = 42, in_dim: int = 8, out_dim: int = 5):
        super().__init__()
        self.num_graphs = num_graphs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_fields = [f"y{i}" for i in range(out_dim)]
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_graphs

    def _make_knn_edges(self, pos: torch.Tensor, k: int = 8) -> torch.Tensor:
        """Construct a simple k-NN graph for demonstration."""
        n = pos.size(0)
        # Brute-force distances for clarity (small synthetic graphs only)
        dists = torch.cdist(pos, pos)
        knn = torch.topk(dists, k + 1, largest=False).indices[:, 1:]
        src = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = knn.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        # Add reverse edges to make it undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index

    def __getitem__(self, idx: int) -> GraphSample:
        self.rng.seed(idx + 17)
        num_nodes = self.rng.randint(120, 200)
        pos = torch.randn(num_nodes, 3)

        # Region heuristics: z>0 shock, near-plane boundary layer, rest free-stream
        shock_mask = (pos[:, 2] > 0.8).float().unsqueeze(-1)
        boundary_mask = (pos[:, 2].abs() < 0.2).float().unsqueeze(-1)
        wake_mask = ((pos[:, 0] > 0.5) & (pos[:, 2].abs() < 0.5)).float().unsqueeze(-1)
        ones = torch.ones_like(shock_mask)
        region_mask = torch.cat([shock_mask, boundary_mask, wake_mask, ones], dim=1)
        region_mask = region_mask / (region_mask.sum(dim=1, keepdim=True) + 1e-6)

        edge_index = self._make_knn_edges(pos, k=12)

        # Input features: simple functions of coordinates + noise
        x = torch.cat(
            [
                pos,
                torch.sin(pos),
                torch.cos(pos[:, :2].sum(dim=1, keepdim=True)),
                torch.randn(num_nodes, self.in_dim - 7),
            ],
            dim=1,
        )

        # Target: synthetic velocity/pressure-like outputs
        base = torch.stack(
            [
                torch.sin(pos[:, 0]) * torch.exp(-pos[:, 2].abs()),
                torch.cos(pos[:, 1]) * torch.exp(-pos[:, 2].abs()),
                torch.tanh(pos[:, 2]),
                pos[:, 0] * 0.1,
                pos[:, 1] * 0.1,
            ],
            dim=1,
        )
        noise = 0.05 * torch.randn(num_nodes, self.out_dim)
        y = base + noise

        # Volume proxy for flux weighting
        volume = torch.full((num_nodes,), 1.0 / num_nodes)
        return GraphSample(
            x=x,
            pos=pos,
            edge_index=edge_index,
            y=y,
            region_mask=region_mask,
            volume=volume,
        )


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_graphs,
    )


class CFDDataset(Dataset):
    """
    Load CFD-exported graphs from node CSV + edge triplet CSV.
    Expects node CSV columns similar to the provided example:
        Node_Number, X__m_, Y__m_, Z__m_, Density__kg_m3_, Eddy_Viscosity__Pa_s_,
        Pressure__Pa_, Static_Enthalpy__J_kg1_, Temperature__K_,
        Velocity_u__m_s1_, Velocity_v__m_s1_, Velocity_w__m_s1_,
        VelocityCurl_X__s1_, VelocityCurl_Y__s1_, VelocityCurl_Z__s1_
    Edges CSV: row_index,col_index,value (value is ignored; indices are 0-based).
    """

    def __init__(
        self,
        node_csvs: Union[str, List[str]],
        edge_csvs: Union[str, List[str]],
        in_fields: Optional[List[str]] = None,
        out_fields: Optional[List[str]] = None,
        auto_region_mask: bool = True,
    ):
        super().__init__()
        if isinstance(node_csvs, str):
            node_csvs = [node_csvs]
        if isinstance(edge_csvs, str):
            edge_csvs = [edge_csvs]
        if len(node_csvs) != len(edge_csvs):
            raise ValueError("node_csvs and edge_csvs must have the same length.")

        default_in = [
            "Density__kg_m3_",
            "Eddy_Viscosity__Pa_s_",
            "Pressure__Pa_",
            "Static_Enthalpy__J_kg1_",
            "Temperature__K_",
        ]
        default_out = [
            "Pressure__Pa_",
            "Velocity_u__m_s1_",
            "Velocity_v__m_s1_",
            "Velocity_w__m_s1_",
            "Density__kg_m3_",
        ]

        self.node_paths = list(node_csvs)
        self.edge_paths = list(edge_csvs)
        self.in_fields = in_fields if in_fields is not None else default_in
        self.out_fields = out_fields if out_fields is not None else default_out
        self.auto_region_mask = auto_region_mask
        self.graphs: List[GraphSample] = []
        total_graphs = len(node_csvs)
        t_load0 = time.perf_counter()
        for idx, (n_path, e_path) in enumerate(zip(node_csvs, edge_csvs), start=1):
            node_table = np.genfromtxt(n_path, delimiter=",", names=True, dtype=np.float32, encoding="utf-8")
            names = node_table.dtype.names

            def fetch_fields(fields: List[str]) -> np.ndarray:
                missing = [f for f in fields if f not in names]
                if missing:
                    raise ValueError(f"Missing columns in {n_path}: {missing}")
                return np.stack([node_table[f] for f in fields], axis=1)

            pos = fetch_fields(["X__m_", "Y__m_", "Z__m_"])
            x = fetch_fields(self.in_fields)
            y = fetch_fields(self.out_fields)

            edge_raw = np.loadtxt(e_path, delimiter=",", skiprows=1, usecols=(0, 1), dtype=np.int64)
            num_nodes = pos.shape[0]
            valid_mask = (
                (edge_raw[:, 0] >= 0)
                & (edge_raw[:, 1] >= 0)
                & (edge_raw[:, 0] < num_nodes)
                & (edge_raw[:, 1] < num_nodes)
            )
            if not np.all(valid_mask):
                dropped = int((~valid_mask).sum())
                total = edge_raw.shape[0]
                print(f"[CFDDataset] Dropped {dropped} / {total} edges exceeding node count {num_nodes} for {n_path}")
            edge_raw = edge_raw[valid_mask]
            edge_index = torch.tensor(edge_raw, dtype=torch.long).t()
            # make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            # remove duplicate edges
            edge_index = torch.unique(edge_index, dim=1)

            volume = torch.full((num_nodes,), 1.0 / num_nodes, dtype=torch.float32)
            region_mask = self._build_region_mask(node_table) if self.auto_region_mask else None
            self.graphs.append(
                GraphSample(
                    x=torch.from_numpy(x),
                    pos=torch.from_numpy(pos),
                    edge_index=edge_index,
                    y=torch.from_numpy(y),
                    region_mask=region_mask,
                    volume=volume,
                )
            )
            if total_graphs >= 10 and (idx == 1 or idx % 5 == 0 or idx == total_graphs):
                dt = time.perf_counter() - t_load0
                print(
                    f"[CFDDataset] loaded {idx}/{total_graphs} graphs "
                    f"(latest={Path(n_path).name}, elapsed={dt:.1f}s)"
                )

        self.in_dim = self.graphs[0].x.shape[1]
        self.out_dim = self.graphs[0].y.shape[1]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> GraphSample:
        return self.graphs[idx]

    @staticmethod
    def _safe_norm(values: np.ndarray) -> np.ndarray:
        values = np.abs(values.astype(np.float32))
        q = float(np.percentile(values, 95.0))
        if not np.isfinite(q) or q < 1e-6:
            q = float(values.mean() + 1e-6)
        return np.clip(values / (q + 1e-8), 0.0, 1.0)

    def _build_region_mask(self, node_table: np.ndarray) -> Optional[torch.Tensor]:
        names = set(node_table.dtype.names or [])
        if "X__m_" not in names:
            return None

        x = node_table["X__m_"].astype(np.float32)
        n = x.shape[0]
        ones = np.ones(n, dtype=np.float32)

        # Shock-like indicator: high vorticity magnitude or pressure deviation.
        if "VelocityCurl_Z__s1_" in names:
            shock = self._safe_norm(node_table["VelocityCurl_Z__s1_"])
        elif "Pressure__Pa_" in names:
            p = node_table["Pressure__Pa_"].astype(np.float32)
            shock = self._safe_norm(p - float(np.median(p)))
        else:
            shock = np.zeros(n, dtype=np.float32)

        # Boundary-layer proxy: high turbulent viscosity.
        if "Eddy_Viscosity__Pa_s_" in names:
            boundary = self._safe_norm(node_table["Eddy_Viscosity__Pa_s_"])
        else:
            boundary = np.zeros(n, dtype=np.float32)

        # Wake proxy: downstream nodes with transverse motion.
        if "Velocity_v__m_s1_" in names:
            v_abs = np.abs(node_table["Velocity_v__m_s1_"].astype(np.float32))
        else:
            v_abs = np.zeros(n, dtype=np.float32)
        x_tail = np.maximum(x - float(np.percentile(x, 60.0)), 0.0)
        wake = self._safe_norm(x_tail * v_abs)

        free = np.clip(ones - np.maximum.reduce([shock, boundary, wake]), 0.0, 1.0)

        weights = np.stack([shock, boundary, wake, free], axis=1) + 1e-6
        weights = weights / weights.sum(axis=1, keepdims=True)
        return torch.from_numpy(weights.astype(np.float32))
