#!/usr/bin/env python3
"""
Simple Synthetic Dataset Generator (for DEC/IDEC/DBSCAN sanity checks)

- Each sample is a 100×100 grid with values in {1,2,3,4}, drawn IID.
- Regimes vary only in the class probabilities (interpretable).
- Features:
    (A) 4-D global histogram of {1,2,3,4}
    (B) 4×4 adjacency co-occurrence (right & down neighbors only) -> 16-D
    (C) Combined 20-D feature vector (histogram + co-occurrence)

Outputs:
- <out_dir>/simple_grid_features.csv
- Optional previews: <out_dir>/grid_example_<regime>.png (one per regime)
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def make_random_grid(n: int = 100, probs=(0.25, 0.25, 0.25, 0.25), rng: np.random.Generator | None = None) -> np.ndarray:
    """IID draws per cell from {1,2,3,4} with given probabilities."""
    if rng is None:
        rng = np.random.default_rng()
    vals = np.array([1, 2, 3, 4], dtype=int)
    grid = rng.choice(vals, size=(n, n), p=probs)
    return grid


def global_histogram_features(grid: np.ndarray) -> np.ndarray:
    """Normalized counts for values 1..4 (shape: (4,))."""
    counts = np.array([(grid == v).sum() for v in [1, 2, 3, 4]], dtype=float)
    counts /= counts.sum()
    return counts


def adjacency_cooccurrence_features(grid: np.ndarray) -> np.ndarray:
    """
    4×4 co-occurrence for 4-neighborhood (right and down only to avoid double-counting).
    Returns a flattened 16-D vector normalized to sum=1.
    """
    # Right pairs: (i,j)->(i,j+1), Down pairs: (i,j)->(i+1,j)
    right_pairs = np.stack([grid[:, :-1], grid[:, 1:]], axis=-1).reshape(-1, 2)
    down_pairs  = np.stack([grid[:-1, :], grid[1:, :]], axis=-1).reshape(-1, 2)
    pairs = np.concatenate([right_pairs, down_pairs], axis=0)

    co = np.zeros((4, 4), dtype=float)
    # Map values {1..4} -> indices {0..3}
    np.add.at(co, (pairs[:, 0] - 1, pairs[:, 1] - 1), 1.0)
    tot = co.sum()
    if tot > 0:
        co /= tot
    return co.reshape(-1)  # 16-D


def default_regimes():
    
    #Interpretable regimes (name, probabilities).
    return [
        ("uniform",       (0.25, 0.25, 0.25, 0.25)),
        ("class1_heavy",  (0.55, 0.15, 0.15, 0.15)),
        ("class2_heavy",  (0.15, 0.55, 0.15, 0.15)),
        ("class3_heavy",  (0.15, 0.15, 0.55, 0.15)),
        ("class4_heavy",  (0.15, 0.15, 0.15, 0.55)),
        ("bimodal_1_3",   (0.40, 0.10, 0.40, 0.10)),
        ("bimodal_1_4",   (0.40, 0.10, 0.10, 0.40)),
        ("bimodal_1_2",   (0.40, 0.40, 0.10, 0.10)),
        ("bimodal_2_3",   (0.10, 0.40, 0.40, 0.10)),
        ("bimodal_2_4",   (0.10, 0.40, 0.10, 0.40)),
        ("bimodal_3_4",   (0.10, 0.10, 0.40, 0.40)),
        ("class1_light",  (0.10, 0.30, 0.30, 0.30)),
        ("class2_light",  (0.30, 0.10, 0.30, 0.30)),
        ("class3_light",  (0.30, 0.30, 0.10, 0.30)),
        ("class4_light",  (0.30, 0.30, 0.30, 0.10)),
    ]


def maybe_save_preview(grid: np.ndarray, path: str):
    """Save a quick image of the grid if matplotlib is available."""
    if not HAS_MPL:
        return
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(grid, interpolation="nearest")
    plt.title(os.path.splitext(os.path.basename(path))[0])
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()


def build_dataset(
    out_dir: str,
    num_samples: int = 60,
    n: int = 100,
    seed: int = 7,
    save_previews: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    regimes = default_regimes()

    # Split samples (evenly as possible) across regimes
    base = num_samples // len(regimes)
    rem = num_samples % len(regimes)
    counts = [base + (1 if i < rem else 0) for i in range(len(regimes))]

    rows = []
    sample_id = 0
    preview_done = set()

    for (name, probs), k in zip(regimes, counts):
        for _ in range(k):
            grid = make_random_grid(n=n, probs=probs, rng=rng)
            h = global_histogram_features(grid)                 # 4-D
            co = adjacency_cooccurrence_features(grid)          # 16-D

            row = {
                "sample_id": sample_id,
                "regime": name,
                "p1": probs[0], "p2": probs[1], "p3": probs[2], "p4": probs[3],
                "h1": h[0], "h2": h[1], "h3": h[2], "h4": h[3],
            }
            for i in range(16):
                row[f"co{i+1:02d}"] = co[i]
            rows.append(row)

            # Save a single preview per regime (optional)
            if save_previews and name not in preview_done:
                preview_path = os.path.join(out_dir, f"grid_example_{name}.png")
                maybe_save_preview(grid, preview_path)
                preview_done.add(name)

            sample_id += 1

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "simple_grid_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote CSV: {csv_path}")
    if save_previews and HAS_MPL:
        print(f"Saved previews in: {out_dir}")
    return df, csv_path


def main():
    parser = argparse.ArgumentParser(description="Generate simple synthetic grids and interpretable features.")
    parser.add_argument("--out_dir", type=str, default="./simple_grids", help="Output directory.")
    parser.add_argument("--num_samples", type=int, default=60, help="Total number of samples to generate.")
    parser.add_argument("--n", type=int, default=100, help="Grid size (n×n).")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed.")
    parser.add_argument("--no_previews", action="store_true", help="Disable saving preview images.")
    args = parser.parse_args()

    build_dataset(
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        n=args.n,
        seed=args.seed,
        save_previews=(not args.no_previews),
    )


if __name__ == "__main__":
    main()
