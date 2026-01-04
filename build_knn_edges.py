import argparse
import numpy as np
from scipy.spatial import cKDTree


def load_positions(node_csv: str) -> np.ndarray:
    data = np.genfromtxt(node_csv, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    required = ["X__m_", "Y__m_", "Z__m_"]
    for col in required:
        if col not in data.dtype.names:
            raise ValueError(f"Missing position column {col} in {node_csv}")
    return np.stack([data["X__m_"], data["Y__m_"], data["Z__m_"]], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build kNN edge CSV from node coordinates.")
    parser.add_argument("--node-csv", required=True, help="Node CSV containing X__m_, Y__m_, Z__m_.")
    parser.add_argument("--out-edge-csv", required=True, help="Output edge CSV (row_index,col_index,value).")
    parser.add_argument("--k", type=int, default=8, help="Number of nearest neighbors per node.")
    parser.add_argument("--workers", type=int, default=-1, help="Number of workers for KDTree query.")
    args = parser.parse_args()

    pos = load_positions(args.node_csv)
    n = pos.shape[0]
    if args.k >= n:
        raise ValueError(f"k ({args.k}) must be smaller than number of nodes ({n})")

    tree = cKDTree(pos)
    try:
        _, idx = tree.query(pos, k=args.k + 1, workers=args.workers)
    except TypeError:
        _, idx = tree.query(pos, k=args.k + 1)

    idx = idx[:, 1:]  # drop self
    src = np.repeat(np.arange(n, dtype=np.int64), args.k)
    dst = idx.reshape(-1).astype(np.int64)

    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    edges = np.stack([a, b], axis=1)
    edges = np.unique(edges, axis=0)

    with open(args.out_edge_csv, "w", encoding="utf-8") as f:
        f.write("row_index,col_index,value\n")
        for s, d in edges:
            f.write(f"{int(s)},{int(d)},1\n")

    print(f"Built {edges.shape[0]} undirected edges (k={args.k}) from {n} nodes.")


if __name__ == "__main__":
    main()
