import argparse
import numpy as np
import meshio
from scipy.spatial import cKDTree


def load_node_csv(path: str):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    names = list(data.dtype.names)
    if "X__m_" in names and "Y__m_" in names and "Z__m_" in names:
        pos = np.stack([data["X__m_"], data["Y__m_"], data["Z__m_"]], axis=1)
    else:
        raise ValueError(f"Position columns X__m_, Y__m_, Z__m_ not found in {path}")
    return data, names, pos


def write_node_csv(path: str, names, data_array):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(names) + "\n")
        for row in data_array:
            vals = []
            for name in names:
                vals.append(row[name])
            f.write(",".join(str(v) for v in vals) + "\n")


def remap_edges(edge_path: str, inverse_map: np.ndarray, out_path: str):
    edge_raw = np.loadtxt(edge_path, delimiter=",", skiprows=1)
    if edge_raw.ndim == 1:
        edge_raw = edge_raw.reshape(1, -1)
    src_old = edge_raw[:, 0].astype(np.int64)
    dst_old = edge_raw[:, 1].astype(np.int64)
    val = edge_raw[:, 2]
    src_new = inverse_map[src_old]
    dst_new = inverse_map[dst_old]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("row_index,col_index,value\n")
        for s, d, v in zip(src_new, dst_new, val):
            f.write(f"{int(s)},{int(d)},{v}\n")


def main():
    parser = argparse.ArgumentParser(description="Reorder node CSV to match CGNS mesh node ordering.")
    parser.add_argument("--cgns", required=True, help="Path to CGNS mesh file.")
    parser.add_argument("--node-csv", required=True, help="Node CSV to reorder.")
    parser.add_argument("--out-csv", required=True, help="Output reordered node CSV.")
    parser.add_argument("--edge-csv", default=None, help="Optional edge CSV to remap (row_index,col_index,value).")
    parser.add_argument("--out-edge-csv", default="edges_remap.csv", help="Output remapped edge CSV.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Max allowed distance for coordinate match.")
    args = parser.parse_args()

    mesh = meshio.read(args.cgns)
    mesh_points = mesh.points
    num_nodes = mesh_points.shape[0]
    print(f"Loaded CGNS mesh with {num_nodes} nodes.")

    node_data, names, pos_src = load_node_csv(args.node_csv)
    if pos_src.shape[0] != node_data.shape[0]:
        raise ValueError("Position array size mismatch with node data.")
    print(f"Loaded CSV with {pos_src.shape[0]} nodes.")

    tree = cKDTree(pos_src)
    dist, idx = tree.query(mesh_points, k=1)
    if (dist > args.tol).any():
        max_d = dist.max()
        raise ValueError(f"Max position mismatch {max_d} exceeds tolerance {args.tol}.")

    # Check uniqueness
    if len(np.unique(idx)) != num_nodes:
        raise ValueError("Non-unique mapping detected; check mesh and CSV coordinates.")

    reordered = node_data[idx]
    # Overwrite Node_Number if present
    if "Node_Number" in names:
        reordered["Node_Number"] = np.arange(num_nodes, dtype=reordered["Node_Number"].dtype)

    write_node_csv(args.out_csv, names, reordered)
    print(f"Reordered node CSV written to {args.out_csv}")

    if args.edge_csv:
        # Build inverse map: old_csv_index -> new_mesh_index
        inverse = np.empty_like(idx)
        inverse[idx] = np.arange(num_nodes, dtype=np.int64)
        remap_edges(args.edge_csv, inverse, args.out_edge_csv)
        print(f"Remapped edge CSV written to {args.out_edge_csv}")


if __name__ == "__main__":
    main()
