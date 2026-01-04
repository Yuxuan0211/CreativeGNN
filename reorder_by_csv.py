import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree


def load_node_csv(path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    names = list(data.dtype.names)
    required = ["X__m_", "Y__m_", "Z__m_"]
    for col in required:
        if col not in names:
            raise ValueError(f"Missing position column {col} in {path}")
    pos = np.stack([data["X__m_"], data["Y__m_"], data["Z__m_"]], axis=1)
    return data, names, pos


def write_node_csv(path: str, names: List[str], data: np.ndarray) -> None:
    cols = [data[name] for name in names]
    out = np.column_stack(cols)
    fmt = ["%d" if name == "Node_Number" else "%.10e" for name in names]
    header = ",".join(names)
    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt=fmt)


def reorder_one(ref_pos: np.ndarray, node_csv: str, out_csv: str, tol: float) -> None:
    data, names, pos = load_node_csv(node_csv)
    if pos.shape[0] != ref_pos.shape[0]:
        raise ValueError(f"Node count mismatch: {node_csv} has {pos.shape[0]} nodes, expected {ref_pos.shape[0]}")

    tree = cKDTree(pos)
    dist, idx = tree.query(ref_pos, k=1)
    if (dist > tol).any():
        max_d = dist.max()
        raise ValueError(f"{node_csv}: max position mismatch {max_d} exceeds tolerance {tol}")
    if len(np.unique(idx)) != ref_pos.shape[0]:
        raise ValueError(f"{node_csv}: non-unique mapping detected, check duplicated coordinates")

    reordered = data[idx]
    if "Node_Number" in names:
        reordered["Node_Number"] = np.arange(ref_pos.shape[0], dtype=reordered["Node_Number"].dtype)

    write_node_csv(out_csv, names, reordered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorder node CSVs to match a reference CSV's node order.")
    parser.add_argument("--ref-csv", required=True, help="Reference node CSV defining target node order.")
    parser.add_argument("--node-csv", default=None, help="Single node CSV to reorder.")
    parser.add_argument("--out-csv", default=None, help="Output CSV for single-file mode.")
    parser.add_argument("--input-dir", default=None, help="Directory containing CSVs to reorder.")
    parser.add_argument("--output-dir", default=None, help="Output directory for batch mode.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for batch mode.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Max allowed distance for coordinate match.")
    args = parser.parse_args()

    ref_data, _, ref_pos = load_node_csv(args.ref_csv)

    if args.node_csv:
        if not args.out_csv:
            raise ValueError("--out-csv is required when using --node-csv")
        reorder_one(ref_pos, args.node_csv, args.out_csv, args.tol)
        print(f"Reordered {args.node_csv} -> {args.out_csv}")
        return

    if args.input_dir:
        if not args.output_dir:
            raise ValueError("--output-dir is required when using --input-dir")
        os.makedirs(args.output_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not files:
            raise ValueError(f"No files matched in {args.input_dir} with pattern {args.pattern}")
        for path in files:
            out_path = os.path.join(args.output_dir, os.path.basename(path))
            reorder_one(ref_pos, path, out_path, args.tol)
            print(f"Reordered {path} -> {out_path}")
        return

    raise ValueError("Provide --node-csv or --input-dir.")


if __name__ == "__main__":
    main()
