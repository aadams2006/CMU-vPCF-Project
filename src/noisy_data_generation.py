#!/usr/bin/env python3
"""Noisy synthetic grid dataset generator.

This module mirrors :mod:`data_generation` but introduces configurable noise to
make the resulting dataset less trivial.  After drawing the baseline grid from
the same categorical regimes, either Gaussian or Poisson noise is injected and
then the standard interpretable features are recomputed on the noisy grid.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from data_generation import (
    adjacency_cooccurrence_features,
    default_regimes,
    global_histogram_features,
    make_random_grid,
    maybe_save_preview,
)


def apply_noise(
    grid: np.ndarray,
    noise_type: str,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a noisy copy of ``grid`` using the requested noise model.

    Parameters
    ----------
    grid:
        The baseline categorical grid with values in ``{1, 2, 3, 4}``.
    noise_type:
        Either ``"gaussian"`` or ``"poisson"``.
    noise_level:
        Controls the magnitude of the perturbation.  For Gaussian noise this is
        interpreted as the standard deviation of the zero-mean noise.  For
        Poisson noise this is a mixing factor in ``[0, 1]`` that interpolates
        between the original grid (``0``) and a fresh Poisson draw (``1``).
    rng:
        NumPy random generator to keep determinism consistent with the rest of
        the pipeline.
    """

    if noise_type not in {"gaussian", "poisson"}:
        raise ValueError(f"Unsupported noise_type={noise_type!r}")

    noise_level = float(noise_level)
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    if noise_type == "gaussian":
        if noise_level == 0:
            return grid.copy()
        noise = rng.normal(loc=0.0, scale=noise_level, size=grid.shape)
        noisy = grid.astype(float) + noise
        # Keep the same categorical support after perturbation.
        noisy = np.clip(noisy, 0.5, 4.5)
        noisy = np.rint(noisy)
        return noisy.astype(int)

    # Poisson branch: interpolate between the original grid and a Poisson draw
    # with rate equal to the underlying categorical value.  When noise_level is
    # zero we simply return the original grid; when it is one the result is the
    # pure Poisson sample.
    if noise_level == 0:
        return grid.copy()
    if noise_level > 1:
        raise ValueError("poisson noise_level must be in [0, 1]")

    lam = np.clip(grid.astype(float), 1e-3, None)
    poisson_sample = rng.poisson(lam=lam)
    mixed = (1.0 - noise_level) * grid + noise_level * poisson_sample
    mixed = np.clip(np.rint(mixed), 1, 4)
    return mixed.astype(int)


def _normalize_noise_types(noise_types: Iterable[str] | str) -> Tuple[str, ...]:
    if isinstance(noise_types, str):
        if noise_types == "both":
            return ("gaussian", "poisson")
        return (noise_types,)

    normalized: list[str] = []
    for noise_type in noise_types:
        if noise_type == "both":
            normalized.extend(["gaussian", "poisson"])
        else:
            normalized.append(noise_type)

    return tuple(dict.fromkeys(normalized))  # remove duplicates while preserving order


def build_dataset(
    out_dir: str,
    num_samples: int = 150,
    n: int = 100,
    seed: int = 7,
    save_previews: bool = True,
    samples_per_regime: int | None = None,
    noise_type: str | Iterable[str] = "both",
    noise_level: float = 0.5,
) -> Dict[str, Tuple[pd.DataFrame, str]]:
    """Generate one dataset per requested noise type and return mapping to CSV paths."""

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    noise_types = _normalize_noise_types(noise_type)
    invalid = [t for t in noise_types if t not in {"gaussian", "poisson"}]
    if invalid:
        raise ValueError(f"Unsupported noise types: {invalid}")
    regimes: list[Tuple[str, Tuple[float, float, float, float]]] = list(default_regimes())

    if samples_per_regime is not None:
        counts = [samples_per_regime] * len(regimes)
        num_samples = samples_per_regime * len(regimes)
    else:
        base = num_samples // len(regimes)
        rem = num_samples % len(regimes)
        counts = [base + (1 if i < rem else 0) for i in range(len(regimes))]

    rows_by_type: dict[str, list[dict[str, float | int | str]]] = {t: [] for t in noise_types}
    sample_ids: dict[str, int] = {t: 0 for t in noise_types}
    preview_done: dict[str, set[str]] = {t: set() for t in noise_types}

    for (name, probs), k in zip(regimes, counts):
        for _ in range(k):
            clean_grid = make_random_grid(n=n, probs=probs, rng=rng)
            for current_noise_type in noise_types:
                noisy_grid = apply_noise(
                    clean_grid,
                    noise_type=current_noise_type,
                    noise_level=noise_level,
                    rng=rng,
                )

                h = global_histogram_features(noisy_grid)
                co = adjacency_cooccurrence_features(noisy_grid)

                row = {
                    "sample_id": sample_ids[current_noise_type],
                    "regime": name,
                    "p1": probs[0],
                    "p2": probs[1],
                    "p3": probs[2],
                    "p4": probs[3],
                    "noise_type": current_noise_type,
                    "noise_level": noise_level,
                    "h1": h[0],
                    "h2": h[1],
                    "h3": h[2],
                    "h4": h[3],
                }
                for i in range(16):
                    row[f"co{i + 1:02d}"] = co[i]
                rows_by_type[current_noise_type].append(row)

                if save_previews and name not in preview_done[current_noise_type]:
                    preview_dir = os.path.join(out_dir, current_noise_type)
                    os.makedirs(preview_dir, exist_ok=True)
                    preview_path = os.path.join(preview_dir, f"grid_example_{name}_{current_noise_type}.png")
                    maybe_save_preview(noisy_grid, preview_path)
                    preview_done[current_noise_type].add(name)

                sample_ids[current_noise_type] += 1

    results: Dict[str, Tuple[pd.DataFrame, str]] = {}
    for current_noise_type, rows in rows_by_type.items():
        noise_dir = os.path.join(out_dir, current_noise_type)
        os.makedirs(noise_dir, exist_ok=True)
        df = pd.DataFrame(rows)
        csv_path = os.path.join(noise_dir, f"noisy_grid_features_{current_noise_type}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Wrote CSV for {current_noise_type}: {csv_path}")
        results[current_noise_type] = (df, csv_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noisy synthetic grids and interpretable features.")
    parser.add_argument("--out_dir", type=str, default="./noisy_simple_grids", help="Output directory.")
    parser.add_argument("--num_samples", type=int, default=150, help="Total number of samples to generate.")
    parser.add_argument(
        "--samples_per_regime",
        type=int,
        default=10,
        help="Number of samples to generate per regime (overrides --num_samples when set).",
    )
    parser.add_argument("--n", type=int, default=100, help="Grid size (n×n).")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed.")
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["gaussian", "poisson", "both"],
        default="both",
        help="Noise model to inject before computing features (or 'both' for both sets).",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.5,
        help="Magnitude of the injected noise (see documentation for semantics).",
    )
    parser.add_argument("--no_previews", action="store_true", help="Disable saving preview images.")
    args = parser.parse_args()

    build_dataset(
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        n=args.n,
        seed=args.seed,
        save_previews=(not args.no_previews),
        samples_per_regime=args.samples_per_regime,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
    )


if __name__ == "__main__":
    main()
