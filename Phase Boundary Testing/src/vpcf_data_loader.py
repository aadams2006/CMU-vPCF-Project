"""Data loading and feature extraction helpers for phase-boundary vPCF training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import hyperspy.api as hs

    HAS_HYPERSPY = True
except ImportError:
    HAS_HYPERSPY = False

try:
    import ncempy.io as nio

    HAS_NCEMPY = True
except ImportError:
    HAS_NCEMPY = False


def read_h5_structure(filename: Union[str, Path]) -> Dict[str, List[str]]:
    """Inspect an HDF5 file and return its group and dataset structure."""
    if not HAS_H5PY:
        raise ImportError("h5py is required to read HDF5 files.")

    structure: Dict[str, List[str]] = {}

    def visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Group):
            structure[name] = []
        elif isinstance(obj, h5py.Dataset):
            parent = "/".join(name.split("/")[:-1]) or "root"
            structure.setdefault(parent, []).append(name.split("/")[-1])

    with h5py.File(filename, "r") as handle:
        structure["root"] = []
        for key in handle.keys():
            if isinstance(handle[key], h5py.Dataset):
                structure["root"].append(key)
        handle.visititems(visitor)

    return structure


def get_h5_frame_count(filename: Union[str, Path]) -> int:
    """Return the number of frames in a vPCF HDF5 file."""
    if not HAS_H5PY:
        raise ImportError("h5py is required to read HDF5 files.")

    with h5py.File(filename, "r") as handle:
        if "experiments" in handle:
            group = handle["experiments"]
            for key in ("vpcf_images", "atomic_positions", "vpcf_origin"):
                if key in group:
                    return int(group[key].shape[0])

        for key in handle.keys():
            if isinstance(handle[key], h5py.Dataset) and handle[key].shape:
                return int(handle[key].shape[0])

    return 0


def load_h5_images_only(
    filename: Union[str, Path],
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Load only the vPCF image stack from an HDF5 file."""
    if not HAS_H5PY:
        raise ImportError("h5py is required to read HDF5 files.")

    with h5py.File(filename, "r") as handle:
        if "experiments" in handle and "vpcf_images" in handle["experiments"]:
            images = handle["experiments"]["vpcf_images"]
        elif "vpcf_images" in handle:
            images = handle["vpcf_images"]
        else:
            raise KeyError("Could not find 'vpcf_images' dataset in HDF5 file.")

        total = int(images.shape[0])
        n_frames = min(total, max_frames) if max_frames else total
        data = np.asarray(images[:n_frames], dtype=np.float32)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print(f"Loaded {n_frames} vPCF images from {filename} with shape {data.shape}")

    return data


def load_dm3_file(filename: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """Load a DM3/DM4 file with hyperspy or ncempy."""
    filename = str(filename)

    if HAS_HYPERSPY:
        signal = hs.load(filename)
        metadata = (
            signal.metadata.as_dictionary()
            if hasattr(signal.metadata, "as_dictionary")
            else {}
        )
        return signal.data, metadata

    if HAS_NCEMPY:
        dm3_data = nio.dm.dmReader(filename)
        return dm3_data["data"], dm3_data.get("metadata", {})

    raise ImportError("Either hyperspy or ncempy is required to read DM3/DM4 files.")


def load_dm3_as_frames(filename: Union[str, Path], verbose: bool = True) -> np.ndarray:
    """Load a DM3/DM4 file as a frame stack."""
    data, _ = load_dm3_file(filename)

    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 4:
        scan_x, scan_y, det_x, det_y = data.shape
        frames = data.reshape(scan_x * scan_y, det_x, det_y)
    elif data.ndim == 3:
        frames = data
    elif data.ndim == 2:
        frames = data[np.newaxis, ...]
    else:
        raise ValueError(f"Unexpected data dimensionality: {data.ndim}")

    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print(f"Loaded DM3 frames from {filename} with shape {frames.shape}")

    return frames


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten image frames into per-sample feature vectors."""
    features = images.reshape(images.shape[0], -1).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_features(features: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize feature vectors with a supported scaling method."""
    features = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if method == "minmax":
        min_vals = features.min(axis=0, keepdims=True)
        max_vals = features.max(axis=0, keepdims=True)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        return (features - min_vals) / ranges

    if method == "standard":
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std == 0] = 1
        return (features - mean) / std

    if method == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return features / norms

    raise ValueError(f"Unknown normalization method: {method}")


def extract_histogram_features(images: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Extract normalized histogram features from images."""
    images = np.nan_to_num(np.asarray(images, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    features = np.zeros((images.shape[0], n_bins), dtype=np.float32)
    bins = np.linspace(images.min(), images.max(), n_bins + 1)

    for idx in range(images.shape[0]):
        hist, _ = np.histogram(images[idx].ravel(), bins=bins)
        hist_sum = hist.sum()
        features[idx] = hist / hist_sum if hist_sum else hist

    return features


def extract_statistical_features(images: np.ndarray) -> np.ndarray:
    """Extract low-dimensional statistical image features."""
    features = np.zeros((images.shape[0], 7), dtype=np.float32)

    for idx in range(images.shape[0]):
        flat = np.asarray(images[idx], dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            continue
        mean = np.mean(flat)
        std = np.std(flat)
        features[idx, 0] = mean
        features[idx, 1] = std
        features[idx, 2] = np.min(flat)
        features[idx, 3] = np.max(flat)
        features[idx, 4] = np.median(flat)
        if std > 0:
            normalized = (flat - mean) / std
            features[idx, 5] = np.mean(normalized**3)
            features[idx, 6] = np.mean(normalized**4) - 3.0

    return features


def downsample_images(images: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample images by average pooling over factor x factor blocks."""
    n_samples, height, width = images.shape
    new_h = height // factor
    new_w = width // factor
    trimmed = images[:, : new_h * factor, : new_w * factor]
    reshaped = trimmed.reshape(n_samples, new_h, factor, new_w, factor)
    return np.nan_to_num(reshaped.mean(axis=(2, 4)).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class VPCFDataset:
    """Container for features and source metadata used by the training pipeline."""

    features: np.ndarray
    raw_images: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    source_file: Optional[str] = None
    sample_names: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.feature_dim = int(self.features.shape[1])
        self.n_samples = int(self.features.shape[0])
        if self.sample_names is None:
            self.sample_names = np.array([f"sample_{idx:06d}" for idx in range(self.n_samples)])

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"VPCFDataset(n_samples={self.n_samples}, "
            f"feature_dim={self.feature_dim}, has_labels={self.labels is not None})"
        )


def _extract_features(images: np.ndarray, feature_method: str) -> np.ndarray:
    if feature_method == "flatten":
        features = flatten_images(images)
    elif feature_method == "histogram":
        features = extract_histogram_features(images)
    elif feature_method == "statistical":
        features = extract_statistical_features(images)
    elif feature_method == "combined":
        features = np.hstack(
            [extract_histogram_features(images), extract_statistical_features(images)]
        )
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")

    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def create_vpcf_dataset_from_h5(
    filename: Union[str, Path],
    feature_method: str = "flatten",
    normalize: Optional[str] = "minmax",
    downsample_factor: Optional[int] = None,
    max_frames: Optional[int] = None,
    keep_raw_images: bool = False,
    verbose: bool = True,
) -> VPCFDataset:
    """Create a VPCFDataset from a vPCF HDF5 file."""
    images = load_h5_images_only(filename, max_frames=max_frames, verbose=verbose)

    if downsample_factor and downsample_factor > 1:
        if verbose:
            print(f"Downsampling H5 images by factor {downsample_factor}")
        images = downsample_images(images, factor=downsample_factor)

    features = _extract_features(images, feature_method)
    if normalize is not None:
        features = normalize_features(features, method=normalize)

    sample_names = np.array([f"h5_frame_{idx:06d}" for idx in range(features.shape[0])])
    return VPCFDataset(
        features=features,
        raw_images=images if keep_raw_images else None,
        labels=None,
        source_file=str(filename),
        sample_names=sample_names,
    )


def create_vpcf_dataset_from_dm3(
    filename: Union[str, Path],
    feature_method: str = "flatten",
    normalize: Optional[str] = "minmax",
    downsample_factor: Optional[int] = None,
    keep_raw_images: bool = False,
    verbose: bool = True,
) -> VPCFDataset:
    """Create a VPCFDataset from a DM3/DM4 file."""
    images = load_dm3_as_frames(filename, verbose=verbose)

    if downsample_factor and downsample_factor > 1:
        if verbose:
            print(f"Downsampling DM3 images by factor {downsample_factor}")
        images = downsample_images(images, factor=downsample_factor)

    features = _extract_features(images, feature_method)
    if normalize is not None:
        features = normalize_features(features, method=normalize)

    sample_names = np.array([f"dm3_frame_{idx:06d}" for idx in range(features.shape[0])])
    return VPCFDataset(
        features=features,
        raw_images=images if keep_raw_images else None,
        labels=None,
        source_file=str(filename),
        sample_names=sample_names,
    )


def load_vpcf_file(filename: Union[str, Path], **kwargs) -> VPCFDataset:
    """Auto-detect and load a supported vPCF or microscopy file."""
    filename = Path(filename)
    ext = filename.suffix.lower()

    if ext in {".h5", ".hdf5", ".hdf"}:
        return create_vpcf_dataset_from_h5(filename, **kwargs)
    if ext in {".dm3", ".dm4"}:
        return create_vpcf_dataset_from_dm3(filename, **kwargs)
    raise ValueError(f"Unsupported file format: {ext}")


def combine_datasets(*datasets: VPCFDataset) -> VPCFDataset:
    """Combine multiple datasets, preserving sample names when possible."""
    if not datasets:
        raise ValueError("At least one dataset is required.")

    feature_dims = {dataset.feature_dim for dataset in datasets}
    if len(feature_dims) != 1:
        raise ValueError(f"Feature dimensions do not match: {sorted(feature_dims)}")

    features = np.vstack([dataset.features for dataset in datasets])
    labels = (
        np.concatenate([dataset.labels for dataset in datasets])
        if all(dataset.labels is not None for dataset in datasets)
        else None
    )
    sample_names = np.concatenate([dataset.sample_names for dataset in datasets])
    source_files = ", ".join(dataset.source_file or "unknown" for dataset in datasets)
    return VPCFDataset(
        features=features,
        raw_images=None,
        labels=labels,
        source_file=source_files,
        sample_names=sample_names,
    )


def check_dependencies() -> Dict[str, bool]:
    """Return availability of optional I/O dependencies."""
    return {
        "h5py": HAS_H5PY,
        "hyperspy": HAS_HYPERSPY,
        "ncempy": HAS_NCEMPY,
    }


def print_file_info(filename: Union[str, Path]) -> None:
    """Print helpful file information before training."""
    filename = Path(filename)
    print(f"File: {filename}")
    print(f"Extension: {filename.suffix.lower()}")
    print(f"Size: {filename.stat().st_size / (1024 * 1024):.2f} MB")

    if filename.suffix.lower() in {".h5", ".hdf5", ".hdf"}:
        print(f"Frame count: {get_h5_frame_count(filename)}")
        structure = read_h5_structure(filename)
        print("HDF5 structure:")
        for group, datasets in structure.items():
            print(f"  {group}/")
            for dataset_name in datasets:
                print(f"    - {dataset_name}")
