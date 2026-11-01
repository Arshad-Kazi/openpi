"""Visualize + compare starting-point distributions of human vs robot demos.

First-frame end-effector position (robot_current_pose[0][:3]) for every episode,
split by source (_human / _robot), plotted in 3D plus 2D projections.
Copies ("(Copy)") are excluded so nothing is double-counted.

Usage:
  python plot_start_dist.py [DATASET_DIR] [--out OUT.png] [--show]
"""
import argparse, glob, os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

DEFAULT_DS = ("/mnt/elements/datasets/sparse_block_picking_with_pause/"
              "datasets/sparse_block_picking_with_pause_0.3_0.7")

def load_starts(ds):
    """Return dict source -> (N,3) array of first-frame xyz."""
    out = {"human": [], "robot": []}
    for e in sorted(glob.glob(os.path.join(ds, "*/episode.h5"))):
        name = os.path.basename(os.path.dirname(e))
        if "(Copy)" in name:
            continue
        src = "human" if "_human" in name else "robot"
        with h5py.File(e, "r") as f:
            out[src].append(f["robot_current_pose"][0, :3])
    return {k: np.array(v) for k, v in out.items() if len(v)}

def summarize(name, P):
    c = P.mean(0)
    print(f"{name:6s} n={len(P):3d}  mean=({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})  "
          f"std=({P.std(0)[0]:.3f}, {P.std(0)[1]:.3f}, {P.std(0)[2]:.3f})")
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=DEFAULT_DS)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "start_dist.png"))
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    data = load_starts(args.dataset)
    assert data, f"No episodes found under {args.dataset}"

    colors = {"human": "#2563eb", "robot": "#dc2626"}
    print("=== starting-position stats (first frame xyz) ===")
    centroids = {k: summarize(k, v) for k, v in data.items()}
    if {"human", "robot"} <= set(data):
        d = np.linalg.norm(centroids["human"] - centroids["robot"])
        print(f"centroid separation (human vs robot): {d*100:.1f} cm")

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Start-position distribution: human vs robot\n{os.path.basename(args.dataset)}",
                 fontsize=13)

    # --- 3D scatter ---
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    for src, P in data.items():
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=18, alpha=0.6,
                   c=colors[src], label=f"{src} (n={len(P)})", depthshade=True)
        c = centroids[src]
        ax.scatter([c[0]], [c[1]], [c[2]], s=200, marker="*",
                   c=colors[src], edgecolors="k", linewidths=1.2)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("3D scatter (★ = centroid)"); ax.legend()

    # --- 2D projections ---
    for i, (a, b, an, bn) in enumerate([(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]):
        ax = fig.add_subplot(2, 2, 2 + i)
        for src, P in data.items():
            ax.scatter(P[:, a], P[:, b], s=18, alpha=0.6, c=colors[src], label=src)
            c = centroids[src]
            ax.scatter([c[a]], [c[b]], s=200, marker="*", c=colors[src],
                       edgecolors="k", linewidths=1.2)
        ax.set_xlabel(f"{an} (m)"); ax.set_ylabel(f"{bn} (m)")
        ax.set_title(f"{an}-{bn} projection"); ax.set_aspect("equal", "datalim")
        ax.grid(True, alpha=0.3); ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140)
    print(f"\nSaved plot -> {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
