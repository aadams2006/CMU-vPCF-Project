"""
Generate cluster-center comparison artifacts for experimental vPCF projects.

For each experimental project under this directory, this script:
1. Finds saved DEC/IDEC clustering runs.
2. Reconstructs cluster-center inputs from the saved latent centroids when
   model weights are available.
3. Builds per-cluster mean vPCFs from the original HDF5 data.
4. Saves side-by-side comparison figures and numeric summaries.

When a run was trained on engineered features (for example 64-bin histograms)
instead of flattened vPCF images, the script cannot decode the centroids into
full vPCF images. In that case it:
- compares decoded center feature profiles against actual cluster-mean features
- picks the nearest real sample in feature space as a proxy image
- compares that proxy vPCF against the actual cluster-mean vPCF
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_HIDDEN_DIMS = [500, 500, 2000]


@dataclass
class RunInfo:
    project_dir: Path
    src_dir: Path
    h5_file: Path
    run_dir: Path
    method: str
    labels: np.ndarray
    centers_latent: np.ndarray
    input_dim: int
    latent_dim: int
    feature_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cluster-center vPCF comparison artifacts."
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Optional project directory name under 'Experimentally-obtained vPCF Testing'.",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["minmax", "none"],
        default="minmax",
        help="Feature normalization assumption used by the saved model.",
    )
    parser.add_argument(
        "--max-projects",
        type=int,
        default=None,
        help="Optional limit for the number of discovered projects to process.",
    )
    return parser.parse_args()


def discover_projects(root: Path, project_name: Optional[str]) -> List[Path]:
    if project_name is not None:
        project_dir = root / project_name
        if not project_dir.is_dir():
            raise FileNotFoundError(f"Project not found: {project_dir}")
        return [project_dir]

    projects = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "data").is_dir() and (child / "results").is_dir() and (child / "src").is_dir():
            projects.append(child)
    return projects


def find_h5_file(project_dir: Path) -> Path:
    h5_files = sorted(project_dir.glob("data/*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 file found in {project_dir / 'data'}")
    return h5_files[0]


def infer_feature_mode(input_dim: int, image_shape: Tuple[int, int]) -> str:
    height, width = image_shape
    if input_dim == height * width:
        return "flatten_image"
    if input_dim == 64:
        return "histogram"
    if input_dim == 7:
        return "statistical"
    if input_dim == 71:
        return "combined"
    side = int(round(math.sqrt(input_dim)))
    if side * side == input_dim:
        return "flatten_square"
    return "unknown"


def load_run_info(project_dir: Path, h5_file: Path) -> List[RunInfo]:
    with h5py.File(h5_file, "r") as f:
        image_shape = tuple(int(v) for v in f["experiments"]["vpcf_images"].shape[1:])

    run_infos: List[RunInfo] = []
    for labels_path in sorted(project_dir.glob("results/**/dec_cluster_labels.csv")):
        run_infos.append(_load_one_run(project_dir, h5_file, image_shape, labels_path, "dec"))
    for labels_path in sorted(project_dir.glob("results/**/idec_cluster_labels.csv")):
        run_infos.append(_load_one_run(project_dir, h5_file, image_shape, labels_path, "idec"))
    return run_infos


def _load_one_run(
    project_dir: Path,
    h5_file: Path,
    image_shape: Tuple[int, int],
    labels_path: Path,
    method: str,
) -> RunInfo:
    run_dir = labels_path.parent
    centers_path = run_dir / f"{method}_cluster_centers.csv"
    if not centers_path.exists():
        raise FileNotFoundError(f"Missing cluster centers file: {centers_path}")

    labels_df = pd.read_csv(labels_path)
    centers_df = pd.read_csv(centers_path)

    center_cols = sorted(
        [col for col in centers_df.columns if col.startswith("dim_")],
        key=lambda name: int(name.split("_")[1]),
    )
    centers_latent = centers_df[center_cols].to_numpy(dtype=np.float32)
    input_dim = _infer_input_dim(run_dir, method)
    latent_dim = centers_latent.shape[1]

    return RunInfo(
        project_dir=project_dir,
        src_dir=project_dir / "src",
        h5_file=h5_file,
        run_dir=run_dir,
        method=method,
        labels=labels_df["cluster"].to_numpy(dtype=np.int32),
        centers_latent=centers_latent,
        input_dim=input_dim,
        latent_dim=latent_dim,
        feature_mode=infer_feature_mode(input_dim, image_shape),
    )


def _infer_input_dim(run_dir: Path, method: str) -> int:
    metrics_path = run_dir / f"{method}_metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        if "n_features" in metrics_df.columns:
            return int(metrics_df.loc[0, "n_features"])

    metadata_path = run_dir / f"{method}_metadata.txt"
    if metadata_path.exists():
        for line in metadata_path.read_text().splitlines():
            if "n_features:" in line:
                return int(line.split(":", 1)[1].strip())

    raise ValueError(f"Could not infer input feature dimension for {run_dir}")


def import_model_classes(src_dir: Path):
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
    from DEC import DEC
    from IDEC import IDEC
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    return DEC, IDEC, Input, Model


def build_decoder(autoencoder, latent_dim: int, keras_input, keras_model):
    decoder_input = keras_input(shape=(latent_dim,), name="decoder_input")
    x = decoder_input
    decoder_layer_indices = sorted(
        {
            int(layer.name.split("_")[1])
            for layer in autoencoder.layers
            if layer.name.startswith("decoder_")
        },
        reverse=True,
    )
    for layer_idx in decoder_layer_indices:
        x = autoencoder.get_layer(name=f"decoder_{layer_idx}")(x)
    return keras_model(inputs=decoder_input, outputs=x, name="decoder")


def load_decoded_centers(run: RunInfo) -> Optional[np.ndarray]:
    try:
        DEC, IDEC, keras_input, keras_model = import_model_classes(run.src_dir)
        dims = [run.input_dim] + DEFAULT_HIDDEN_DIMS + [run.latent_dim]

        if run.method == "dec":
            model = DEC(dims=dims, n_clusters=run.latent_dim, save_dir=str(run.run_dir))
            ae_path = run.run_dir / "ae_weights.weights.h5"
            final_path = run.run_dir / "DEC_model_final.weights.h5"
            if ae_path.exists():
                model.autoencoder.load_weights(str(ae_path))
            if final_path.exists():
                model.model.load_weights(str(final_path))
            decoder = build_decoder(model.autoencoder, run.latent_dim, keras_input, keras_model)
        else:
            model = IDEC(dims=dims, n_clusters=run.latent_dim, save_dir=str(run.run_dir))
            ae_path = run.run_dir / "ae_weights.weights.h5"
            final_path = run.run_dir / "IDEC_model_final.weights.h5"
            if ae_path.exists():
                model.autoencoder.load_weights(str(ae_path))
            if final_path.exists():
                model.model.load_weights(str(final_path))
            decoder = build_decoder(model.autoencoder, run.latent_dim, keras_input, keras_model)

        return decoder.predict(run.centers_latent, verbose=0)
    except Exception as exc:
        print(f"  Warning: could not decode centers for {run.run_dir}: {exc}")
        return None


def project_global_minmax(h5_file: Path) -> Tuple[float, float]:
    global_min = np.inf
    global_max = -np.inf
    with h5py.File(h5_file, "r") as f:
        images = f["experiments"]["vpcf_images"]
        for idx in range(images.shape[0]):
            image = images[idx]
            img_min = float(np.min(image))
            img_max = float(np.max(image))
            if img_min < global_min:
                global_min = img_min
            if img_max > global_max:
                global_max = img_max
    return global_min, global_max


def extract_histogram_feature(image: np.ndarray, bins: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(image.reshape(-1), bins=bins)
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_statistical_feature(image: np.ndarray) -> np.ndarray:
    flat = image.reshape(-1).astype(np.float32)
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    result = np.zeros(7, dtype=np.float32)
    result[0] = mean
    result[1] = std
    result[2] = float(np.min(flat))
    result[3] = float(np.max(flat))
    result[4] = float(np.median(flat))
    if std > 0:
        normalized = (flat - mean) / std
        result[5] = float(np.mean(normalized ** 3))
        result[6] = float(np.mean(normalized ** 4) - 3.0)
    return result


def extract_feature(image: np.ndarray, feature_mode: str, bins: Optional[np.ndarray]) -> np.ndarray:
    if feature_mode == "histogram":
        if bins is None:
            raise ValueError("Histogram bins are required for histogram features")
        return extract_histogram_feature(image, bins)
    if feature_mode == "statistical":
        return extract_statistical_feature(image)
    if feature_mode == "combined":
        if bins is None:
            raise ValueError("Histogram bins are required for combined features")
        return np.concatenate([extract_histogram_feature(image, bins), extract_statistical_feature(image)])
    raise ValueError(f"Feature extraction not supported for mode '{feature_mode}'")


def normalize_features(features: np.ndarray, normalize_mode: str) -> np.ndarray:
    if normalize_mode == "none":
        return features
    min_vals = features.min(axis=0, keepdims=True)
    max_vals = features.max(axis=0, keepdims=True)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    return (features - min_vals) / ranges


def save_feature_tables(output_dir: Path, decoded_features: np.ndarray, mean_features: np.ndarray) -> None:
    decoded_df = pd.DataFrame(decoded_features, columns=[f"dim_{i}" for i in range(decoded_features.shape[1])])
    decoded_df.insert(0, "cluster", np.arange(len(decoded_df)))
    decoded_df.to_csv(output_dir / "decoded_center_features.csv", index=False)

    mean_df = pd.DataFrame(mean_features, columns=[f"dim_{i}" for i in range(mean_features.shape[1])])
    mean_df.insert(0, "cluster", np.arange(len(mean_df)))
    mean_df.to_csv(output_dir / "actual_cluster_mean_features.csv", index=False)


def save_mean_images(output_dir: Path, mean_images: np.ndarray) -> None:
    arrays = {f"cluster_{cluster_id}": mean_images[cluster_id].astype(np.float32) for cluster_id in range(mean_images.shape[0])}
    np.savez_compressed(output_dir / "actual_cluster_mean_vpcfs.npz", **arrays)


def plot_feature_profiles(output_dir: Path, decoded_features: np.ndarray, mean_features: np.ndarray, method: str) -> None:
    n_clusters = decoded_features.shape[0]
    fig, axes = plt.subplots(n_clusters, 1, figsize=(14, max(3.5, 2.8 * n_clusters)), squeeze=False)
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id, 0]
        ax.plot(decoded_features[cluster_id], label="Decoded center feature", linewidth=2.0)
        ax.plot(mean_features[cluster_id], label="Actual cluster-mean feature", linewidth=1.5)
        ax.set_title(f"{method.upper()} cluster {cluster_id} feature profile")
        ax.set_xlabel("Feature dimension")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "cluster_center_feature_profiles.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_image_gallery(
    output_dir: Path,
    method: str,
    title_suffix: str,
    left_images: np.ndarray,
    middle_images: np.ndarray,
    left_label: str,
    middle_label: str,
) -> None:
    n_clusters = left_images.shape[0]
    fig, axes = plt.subplots(n_clusters, 3, figsize=(12, max(4.0, 3.3 * n_clusters)), squeeze=False)
    for cluster_id in range(n_clusters):
        left = left_images[cluster_id]
        middle = middle_images[cluster_id]
        diff = np.abs(left - middle)

        vmin = min(float(np.min(left)), float(np.min(middle)))
        vmax = max(float(np.max(left)), float(np.max(middle)))

        panels = [
            (left, left_label),
            (middle, middle_label),
            (diff, "Absolute difference"),
        ]
        for col_idx, (image, label) in enumerate(panels):
            ax = axes[cluster_id, col_idx]
            if col_idx < 2:
                im = ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(image, cmap="magma")
            ax.set_title(f"Cluster {cluster_id}: {label}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{method.upper()} cluster comparison: {title_suffix}", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "cluster_center_vpcf_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    output_dir: Path,
    method: str,
    feature_mode: str,
    image_comparison_mode: str,
    counts: np.ndarray,
    notes: List[str],
) -> None:
    lines = [
        f"Method: {method.upper()}",
        f"Feature mode inferred from saved run: {feature_mode}",
        f"Image comparison mode: {image_comparison_mode}",
        "",
        "Cluster sample counts:",
    ]
    for cluster_id, count in enumerate(counts):
        lines.append(f"  Cluster {cluster_id}: {int(count)}")
    if notes:
        lines.extend(["", "Notes:"])
        for note in notes:
            lines.append(f"  - {note}")
    (output_dir / "comparison_summary.txt").write_text("\n".join(lines))


def process_project(project_dir: Path, normalize_mode: str) -> None:
    print(f"\nProcessing project: {project_dir.name}")
    h5_file = find_h5_file(project_dir)
    run_infos = load_run_info(project_dir, h5_file)
    if not run_infos:
        print("  No DEC/IDEC runs found. Skipping.")
        return

    image_shape: Tuple[int, int]
    with h5py.File(h5_file, "r") as f:
        images = f["experiments"]["vpcf_images"]
        n_samples = int(images.shape[0])
        image_shape = (int(images.shape[1]), int(images.shape[2]))

    needs_hist_bins = any(run.feature_mode in {"histogram", "combined"} for run in run_infos)
    global_min = global_max = None
    if needs_hist_bins:
        print("  Scanning image range for histogram features...")
        global_min, global_max = project_global_minmax(h5_file)
        print(f"  Image range: [{global_min:.6f}, {global_max:.6f}]")
    bins = np.linspace(global_min, global_max, 65) if needs_hist_bins else None

    decoded_by_run: Dict[Path, Optional[np.ndarray]] = {}
    for run in run_infos:
        print(f"  Loading decoder for {run.method.upper()} run: {run.run_dir.relative_to(project_dir)}")
        decoded_by_run[run.run_dir] = load_decoded_centers(run)

    feature_specs = {
        (run.feature_mode, run.input_dim)
        for run in run_infos
        if run.feature_mode in {"histogram", "statistical", "combined"}
    }
    feature_matrices: Dict[Tuple[str, int], np.ndarray] = {
        spec: np.zeros((n_samples, spec[1]), dtype=np.float32)
        for spec in feature_specs
    }

    image_sums: Dict[Path, np.ndarray] = {}
    counts_by_run: Dict[Path, np.ndarray] = {}
    for run in run_infos:
        n_clusters = run.centers_latent.shape[0]
        image_sums[run.run_dir] = np.zeros((n_clusters, image_shape[0], image_shape[1]), dtype=np.float64)
        counts_by_run[run.run_dir] = np.zeros(n_clusters, dtype=np.int64)

    print("  Building cluster mean vPCFs and raw features...")
    with h5py.File(h5_file, "r") as f:
        images = f["experiments"]["vpcf_images"]
        for sample_idx in range(n_samples):
            image = images[sample_idx].astype(np.float32)

            for spec in feature_specs:
                feature_mode, _ = spec
                feature_matrices[spec][sample_idx] = extract_feature(image, feature_mode, bins)

            for run in run_infos:
                cluster_id = int(run.labels[sample_idx])
                image_sums[run.run_dir][cluster_id] += image
                counts_by_run[run.run_dir][cluster_id] += 1

            if (sample_idx + 1) % 250 == 0 or sample_idx + 1 == n_samples:
                print(f"    Processed {sample_idx + 1}/{n_samples} samples")

    normalized_features: Dict[Tuple[str, int], np.ndarray] = {
        spec: normalize_features(matrix, normalize_mode)
        for spec, matrix in feature_matrices.items()
    }

    for run in run_infos:
        print(f"  Saving comparison outputs for {run.method.upper()} run: {run.run_dir.relative_to(project_dir)}")
        output_dir = run.run_dir / "center_vpcf_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)

        counts = counts_by_run[run.run_dir]
        mean_images = image_sums[run.run_dir] / counts[:, None, None]
        save_mean_images(output_dir, mean_images)

        decoded_features = decoded_by_run[run.run_dir]
        notes: List[str] = []
        metrics_rows: List[Dict[str, object]] = []

        feature_matrix = normalized_features.get((run.feature_mode, run.input_dim))
        actual_mean_features = None
        if decoded_features is not None and feature_matrix is not None:
            actual_mean_features = np.zeros_like(decoded_features, dtype=np.float32)
            for cluster_id in range(decoded_features.shape[0]):
                cluster_mask = run.labels == cluster_id
                actual_mean_features[cluster_id] = feature_matrix[cluster_mask].mean(axis=0)
            save_feature_tables(output_dir, decoded_features, actual_mean_features)
            plot_feature_profiles(output_dir, decoded_features, actual_mean_features, run.method)

        if run.feature_mode == "flatten_image" and decoded_features is not None:
            center_images = decoded_features.reshape((-1, image_shape[0], image_shape[1]))
            plot_image_gallery(
                output_dir,
                run.method,
                "decoded center image vs actual cluster mean vPCF",
                center_images,
                mean_images,
                "Decoded center vPCF",
                "Actual cluster-mean vPCF",
            )
            for cluster_id in range(center_images.shape[0]):
                diff = center_images[cluster_id] - mean_images[cluster_id]
                metrics_rows.append(
                    {
                        "cluster": cluster_id,
                        "count": int(counts[cluster_id]),
                        "comparison_mode": "decoded_image",
                        "image_mae": float(np.mean(np.abs(diff))),
                        "image_mse": float(np.mean(diff ** 2)),
                    }
                )
        else:
            if run.feature_mode != "flatten_image":
                notes.append(
                    "This run was trained on engineered features rather than raw flattened vPCF images, "
                    "so the saved centroids cannot be decoded into full vPCF images directly."
                )

            if decoded_features is None or feature_matrix is None:
                notes.append("Decoder or feature matrix unavailable; only actual cluster-mean vPCFs were saved.")
                for cluster_id in range(len(counts)):
                    metrics_rows.append(
                        {
                            "cluster": cluster_id,
                            "count": int(counts[cluster_id]),
                            "comparison_mode": "mean_only",
                        }
                    )
            else:
                proxy_indices = []
                with h5py.File(h5_file, "r") as f:
                    images = f["experiments"]["vpcf_images"]
                    proxy_images = np.zeros_like(mean_images, dtype=np.float32)
                    for cluster_id in range(decoded_features.shape[0]):
                        cluster_indices = np.where(run.labels == cluster_id)[0]
                        cluster_features = feature_matrix[cluster_indices]
                        distances = np.linalg.norm(cluster_features - decoded_features[cluster_id], axis=1)
                        best_local_idx = int(np.argmin(distances))
                        best_sample_idx = int(cluster_indices[best_local_idx])
                        proxy_indices.append(best_sample_idx)
                        proxy_images[cluster_id] = images[best_sample_idx].astype(np.float32)

                        feature_diff = decoded_features[cluster_id] - actual_mean_features[cluster_id]
                        image_diff = proxy_images[cluster_id] - mean_images[cluster_id]
                        metrics_rows.append(
                            {
                                "cluster": cluster_id,
                                "count": int(counts[cluster_id]),
                                "comparison_mode": "feature_proxy_nearest_sample",
                                "proxy_sample_idx": best_sample_idx,
                                "proxy_feature_distance": float(distances[best_local_idx]),
                                "feature_mae": float(np.mean(np.abs(feature_diff))),
                                "feature_mse": float(np.mean(feature_diff ** 2)),
                                "image_proxy_mae": float(np.mean(np.abs(image_diff))),
                                "image_proxy_mse": float(np.mean(image_diff ** 2)),
                            }
                        )

                pd.DataFrame(
                    {
                        "cluster": np.arange(len(proxy_indices)),
                        "proxy_sample_idx": proxy_indices,
                    }
                ).to_csv(output_dir / "proxy_sample_selection.csv", index=False)

                plot_image_gallery(
                    output_dir,
                    run.method,
                    "nearest-sample proxy vs actual cluster mean vPCF",
                    proxy_images,
                    mean_images,
                    "Nearest real sample to decoded center",
                    "Actual cluster-mean vPCF",
                )

        pd.DataFrame(metrics_rows).to_csv(output_dir / "cluster_comparison_metrics.csv", index=False)
        write_summary(
            output_dir=output_dir,
            method=run.method,
            feature_mode=run.feature_mode,
            image_comparison_mode=metrics_rows[0]["comparison_mode"] if metrics_rows else "unknown",
            counts=counts,
            notes=notes,
        )


def main() -> None:
    args = parse_args()
    projects = discover_projects(ROOT, args.project)
    if args.max_projects is not None:
        projects = projects[: args.max_projects]

    if not projects:
        raise SystemExit("No experimental projects found.")

    for project_dir in projects:
        process_project(project_dir, normalize_mode=args.normalize)

    print("\nDone.")


if __name__ == "__main__":
    main()
