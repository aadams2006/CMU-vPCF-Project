"""
Data loading and preprocessing utilities for vPCF HDF5 and DM3 files.

This module provides functions to:
- Load vPCF data from HDF5 (.h5) and Digital Micrograph (.dm3) files
- Extract and preprocess features for DEC/IDEC clustering models
- Create feature matrices suitable for training

Supported file formats:
- HDF5 (.h5): Contains experiments with atomic positions, vPCF images, origins, and peaks
- DM3 (.dm3): Digital Micrograph format (requires hyperspy or ncempy)
"""

from __future__ import annotations

import os
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


# =============================================================================
# HDF5 Data Loading Functions
# =============================================================================

def read_h5_structure(filename: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Inspect the structure of an HDF5 file and return group/dataset names.
    
    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file.
        
    Returns
    -------
    dict
        Dictionary mapping group names to lists of dataset names.
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required to read HDF5 files. Install with: pip install h5py")
    
    structure = {}
    
    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            structure[name] = []
        elif isinstance(obj, h5py.Dataset):
            parent = "/".join(name.split("/")[:-1]) or "root"
            if parent not in structure:
                structure[parent] = []
            structure[parent].append(name.split("/")[-1])
    
    with h5py.File(filename, 'r') as f:
        # Handle root datasets
        structure["root"] = []
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                structure["root"].append(key)
        f.visititems(visitor)
    
    return structure


def get_h5_frame_count(filename: Union[str, Path]) -> int:
    """
    Get the number of frames/samples in an HDF5 vPCF file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file.
        
    Returns
    -------
    int
        Number of frames in the dataset.
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required. Install with: pip install h5py")
    
    with h5py.File(filename, 'r') as f:
        if "experiments" in f:
            grp = f["experiments"]
            # Try common dataset names to determine frame count
            for key in ["vpcf_images", "atomic_positions", "vpcf_origin"]:
                if key in grp:
                    return grp[key].shape[0]
        # Fallback: check root level
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset) and len(f[key].shape) > 0:
                return f[key].shape[0]
    
    return 0


def read_vpcf_frame(filename: Union[str, Path], frame_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read one frame (atomic_position, origin, peaks, image) from an HDF5 vPCF file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file.
    frame_index : int
        Index of the frame to read.
        
    Returns
    -------
    tuple
        (atomic_position, vpcf_origin, vpcf_peaks, vpcf_image)
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required. Install with: pip install h5py")
    
    with h5py.File(filename, 'r') as f:
        grp = f["experiments"]
        
        atomic_position = grp["atomic_positions"][frame_index]
        vpcf_origin = grp["vpcf_origin"][frame_index]
        
        flat_peaks = grp["vpcf_peaks"][frame_index]
        peaks_shape = grp["peaks_shapes"][frame_index]
        vpcf_peaks = flat_peaks.reshape(peaks_shape)
        
        vpcf_image = grp["vpcf_images"][frame_index]
    
    return atomic_position, vpcf_origin, vpcf_peaks, vpcf_image


def load_all_h5_frames(
    filename: Union[str, Path],
    max_frames: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, List[np.ndarray]]:
    """
    Load all frames from an HDF5 vPCF file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file.
    max_frames : int, optional
        Maximum number of frames to load. If None, load all.
    verbose : bool
        Whether to print progress information.
        
    Returns
    -------
    dict
        Dictionary containing lists of atomic_positions, vpcf_origins, 
        vpcf_peaks, and vpcf_images.
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required. Install with: pip install h5py")
    
    total_frames = get_h5_frame_count(filename)
    n_frames = min(total_frames, max_frames) if max_frames else total_frames
    
    if verbose:
        print(f"Loading {n_frames} frames from {filename}...")
    
    data = {
        "atomic_positions": [],
        "vpcf_origins": [],
        "vpcf_peaks": [],
        "vpcf_images": []
    }
    
    for i in range(n_frames):
        atomic_pos, origin, peaks, image = read_vpcf_frame(filename, i)
        data["atomic_positions"].append(atomic_pos)
        data["vpcf_origins"].append(origin)
        data["vpcf_peaks"].append(peaks)
        data["vpcf_images"].append(image)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{n_frames} frames...")
    
    if verbose:
        print(f"  Done! Loaded {n_frames} frames.")
    
    return data


def load_h5_images_only(
    filename: Union[str, Path],
    max_frames: Optional[int] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Efficiently load only vPCF images from an HDF5 file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file.
    max_frames : int, optional
        Maximum number of frames to load.
    verbose : bool
        Whether to print progress information.
        
    Returns
    -------
    np.ndarray
        Array of vPCF images with shape (n_frames, height, width).
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required. Install with: pip install h5py")
    
    with h5py.File(filename, 'r') as f:
        if "experiments" in f and "vpcf_images" in f["experiments"]:
            images = f["experiments"]["vpcf_images"]
        elif "vpcf_images" in f:
            images = f["vpcf_images"]
        else:
            raise KeyError("Could not find 'vpcf_images' dataset in HDF5 file.")
        
        total = images.shape[0]
        n_frames = min(total, max_frames) if max_frames else total
        
        if verbose:
            print(f"Loading {n_frames} vPCF images from {filename}...")
        
        data = images[:n_frames]
    
    if verbose:
        print(f"  Loaded images with shape: {data.shape}")
    
    return data


# =============================================================================
# DM3 Data Loading Functions
# =============================================================================

def load_dm3_file(filename: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """
    Load data from a Digital Micrograph (.dm3) file.
    
    Uses hyperspy if available, otherwise falls back to ncempy.
    
    Parameters
    ----------
    filename : str or Path
        Path to the DM3 file.
        
    Returns
    -------
    tuple
        (data_array, metadata_dict)
    """
    filename = str(filename)
    
    if HAS_HYPERSPY:
        signal = hs.load(filename)
        data = signal.data
        metadata = signal.metadata.as_dictionary() if hasattr(signal.metadata, 'as_dictionary') else {}
        return data, metadata
    
    elif HAS_NCEMPY:
        dm3_data = nio.dm.dmReader(filename)
        data = dm3_data['data']
        metadata = dm3_data.get('metadata', {})
        return data, metadata
    
    else:
        raise ImportError(
            "Either hyperspy or ncempy is required to read DM3 files.\n"
            "Install with: pip install hyperspy\n"
            "Or: pip install ncempy"
        )


def load_dm3_as_frames(
    filename: Union[str, Path],
    verbose: bool = True
) -> np.ndarray:
    """
    Load a DM3 file and return it as a stack of frames.
    
    Parameters
    ----------
    filename : str or Path
        Path to the DM3 file.
    verbose : bool
        Whether to print information about the loaded data.
        
    Returns
    -------
    np.ndarray
        Data array. If 4D (scan_x, scan_y, det_x, det_y), returns as 
        (scan_x * scan_y, det_x, det_y) frames.
    """
    data, metadata = load_dm3_file(filename)
    
    if verbose:
        print(f"Loaded DM3 file: {filename}")
        print(f"  Original shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
    
    # Handle different dimensionalities
    if data.ndim == 4:
        # 4D-STEM data: (scan_x, scan_y, det_x, det_y)
        scan_x, scan_y, det_x, det_y = data.shape
        frames = data.reshape(scan_x * scan_y, det_x, det_y)
        if verbose:
            print(f"  Reshaped to: {frames.shape} (flattened scan dimensions)")
    elif data.ndim == 3:
        # Already a stack of frames
        frames = data
    elif data.ndim == 2:
        # Single image - add frame dimension
        frames = data[np.newaxis, ...]
    else:
        raise ValueError(f"Unexpected data dimensionality: {data.ndim}")
    
    return frames


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def flatten_images(images: np.ndarray) -> np.ndarray:
    """
    Flatten 2D/3D images to 1D feature vectors.
    
    Parameters
    ----------
    images : np.ndarray
        Array of shape (n_samples, height, width) or (n_samples, height, width, channels)
        
    Returns
    -------
    np.ndarray
        Flattened features of shape (n_samples, n_features)
    """
    n_samples = images.shape[0]
    return images.reshape(n_samples, -1).astype(np.float32)


def normalize_features(features: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize feature matrix.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    method : str
        Normalization method: "minmax", "standard", or "l2"
        
    Returns
    -------
    np.ndarray
        Normalized features.
    """
    if method == "minmax":
        min_vals = features.min(axis=0, keepdims=True)
        max_vals = features.max(axis=0, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        return (features - min_vals) / range_vals
    
    elif method == "standard":
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        return (features - mean) / std
    
    elif method == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return features / norms
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_histogram_features(images: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """
    Extract histogram features from images.
    
    Parameters
    ----------
    images : np.ndarray
        Array of images with shape (n_samples, height, width)
    n_bins : int
        Number of histogram bins.
        
    Returns
    -------
    np.ndarray
        Histogram features of shape (n_samples, n_bins)
    """
    n_samples = images.shape[0]
    features = np.zeros((n_samples, n_bins), dtype=np.float32)
    
    # Compute global min/max for consistent binning
    global_min = images.min()
    global_max = images.max()
    bins = np.linspace(global_min, global_max, n_bins + 1)
    
    for i in range(n_samples):
        hist, _ = np.histogram(images[i].flatten(), bins=bins)
        features[i] = hist / hist.sum()  # Normalize to probability
    
    return features


def extract_statistical_features(images: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from images.
    
    Parameters
    ----------
    images : np.ndarray
        Array of images with shape (n_samples, height, width)
        
    Returns
    -------
    np.ndarray
        Statistical features of shape (n_samples, 7):
        [mean, std, min, max, median, skewness, kurtosis]
    """
    n_samples = images.shape[0]
    features = np.zeros((n_samples, 7), dtype=np.float32)
    
    for i in range(n_samples):
        flat = images[i].flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        
        features[i, 0] = mean
        features[i, 1] = std
        features[i, 2] = np.min(flat)
        features[i, 3] = np.max(flat)
        features[i, 4] = np.median(flat)
        
        # Skewness and kurtosis
        if std > 0:
            normalized = (flat - mean) / std
            features[i, 5] = np.mean(normalized ** 3)  # Skewness
            features[i, 6] = np.mean(normalized ** 4) - 3  # Excess kurtosis
    
    return features


def downsample_images(images: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsample images by a factor using average pooling.
    
    Parameters
    ----------
    images : np.ndarray
        Array of images with shape (n_samples, height, width)
    factor : int
        Downsampling factor.
        
    Returns
    -------
    np.ndarray
        Downsampled images.
    """
    n_samples, height, width = images.shape
    new_h = height // factor
    new_w = width // factor
    
    # Trim to be divisible by factor
    trimmed = images[:, :new_h * factor, :new_w * factor]
    
    # Reshape and average
    reshaped = trimmed.reshape(n_samples, new_h, factor, new_w, factor)
    downsampled = reshaped.mean(axis=(2, 4)).astype(np.float32)
    
    return downsampled


# =============================================================================
# Pipeline Functions
# =============================================================================

class VPCFDataset:
    """
    Container class for vPCF dataset with features prepared for DEC/IDEC training.
    
    Attributes
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    raw_images : np.ndarray
        Original images (optional, may be None to save memory)
    labels : np.ndarray
        Ground truth labels if available (may be None)
    source_file : str
        Path to the source file
    feature_dim : int
        Dimensionality of features
    """
    
    def __init__(
        self,
        features: np.ndarray,
        raw_images: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        source_file: Optional[str] = None
    ):
        self.features = features
        self.raw_images = raw_images
        self.labels = labels
        self.source_file = source_file
        self.feature_dim = features.shape[1]
        self.n_samples = features.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __repr__(self):
        return (
            f"VPCFDataset(n_samples={self.n_samples}, "
            f"feature_dim={self.feature_dim}, "
            f"has_labels={self.labels is not None})"
        )


def create_vpcf_dataset_from_h5(
    filename: Union[str, Path],
    feature_method: str = "flatten",
    normalize: str = "minmax",
    downsample_factor: Optional[int] = None,
    max_frames: Optional[int] = None,
    keep_raw_images: bool = False,
    verbose: bool = True
) -> VPCFDataset:
    """
    Create a VPCFDataset from an HDF5 file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file.
    feature_method : str
        Feature extraction method: "flatten", "histogram", "statistical", or "combined"
    normalize : str
        Normalization method: "minmax", "standard", "l2", or None
    downsample_factor : int, optional
        If provided, downsample images before feature extraction.
    max_frames : int, optional
        Maximum number of frames to load.
    keep_raw_images : bool
        Whether to keep raw images in the dataset object.
    verbose : bool
        Whether to print progress information.
        
    Returns
    -------
    VPCFDataset
        Dataset ready for DEC/IDEC training.
    """
    # Load images
    images = load_h5_images_only(filename, max_frames=max_frames, verbose=verbose)
    
    # Downsample if requested
    if downsample_factor is not None and downsample_factor > 1:
        if verbose:
            print(f"Downsampling images by factor {downsample_factor}...")
        images = downsample_images(images, factor=downsample_factor)
        if verbose:
            print(f"  New shape: {images.shape}")
    
    # Extract features
    if verbose:
        print(f"Extracting features using method: {feature_method}")
    
    if feature_method == "flatten":
        features = flatten_images(images)
    elif feature_method == "histogram":
        features = extract_histogram_features(images)
    elif feature_method == "statistical":
        features = extract_statistical_features(images)
    elif feature_method == "combined":
        hist_features = extract_histogram_features(images)
        stat_features = extract_statistical_features(images)
        features = np.hstack([hist_features, stat_features])
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")
    
    if verbose:
        print(f"  Feature shape: {features.shape}")
    
    # Normalize
    if normalize is not None:
        if verbose:
            print(f"Normalizing features using method: {normalize}")
        features = normalize_features(features, method=normalize)
    
    return VPCFDataset(
        features=features,
        raw_images=images if keep_raw_images else None,
        labels=None,
        source_file=str(filename)
    )


def create_vpcf_dataset_from_dm3(
    filename: Union[str, Path],
    feature_method: str = "flatten",
    normalize: str = "minmax",
    downsample_factor: Optional[int] = None,
    keep_raw_images: bool = False,
    verbose: bool = True
) -> VPCFDataset:
    """
    Create a VPCFDataset from a DM3 file.
    
    Parameters
    ----------
    filename : str or Path
        Path to the DM3 file.
    feature_method : str
        Feature extraction method: "flatten", "histogram", "statistical", or "combined"
    normalize : str
        Normalization method: "minmax", "standard", "l2", or None
    downsample_factor : int, optional
        If provided, downsample images before feature extraction.
    keep_raw_images : bool
        Whether to keep raw images in the dataset object.
    verbose : bool
        Whether to print progress information.
        
    Returns
    -------
    VPCFDataset
        Dataset ready for DEC/IDEC training.
    """
    # Load images
    images = load_dm3_as_frames(filename, verbose=verbose)
    
    # Downsample if requested
    if downsample_factor is not None and downsample_factor > 1:
        if verbose:
            print(f"Downsampling images by factor {downsample_factor}...")
        images = downsample_images(images, factor=downsample_factor)
        if verbose:
            print(f"  New shape: {images.shape}")
    
    # Extract features
    if verbose:
        print(f"Extracting features using method: {feature_method}")
    
    if feature_method == "flatten":
        features = flatten_images(images)
    elif feature_method == "histogram":
        features = extract_histogram_features(images)
    elif feature_method == "statistical":
        features = extract_statistical_features(images)
    elif feature_method == "combined":
        hist_features = extract_histogram_features(images)
        stat_features = extract_statistical_features(images)
        features = np.hstack([hist_features, stat_features])
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")
    
    if verbose:
        print(f"  Feature shape: {features.shape}")
    
    # Normalize
    if normalize is not None:
        if verbose:
            print(f"Normalizing features using method: {normalize}")
        features = normalize_features(features, method=normalize)
    
    return VPCFDataset(
        features=features,
        raw_images=images if keep_raw_images else None,
        labels=None,
        source_file=str(filename)
    )


def load_vpcf_file(
    filename: Union[str, Path],
    **kwargs
) -> VPCFDataset:
    """
    Auto-detect file type and load vPCF data.
    
    Parameters
    ----------
    filename : str or Path
        Path to the data file (.h5 or .dm3)
    **kwargs
        Additional arguments passed to the appropriate loader function.
        
    Returns
    -------
    VPCFDataset
        Dataset ready for DEC/IDEC training.
    """
    filename = Path(filename)
    ext = filename.suffix.lower()
    
    if ext in ['.h5', '.hdf5', '.hdf']:
        return create_vpcf_dataset_from_h5(filename, **kwargs)
    elif ext in ['.dm3', '.dm4']:
        return create_vpcf_dataset_from_dm3(filename, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .h5, .hdf5, .dm3, .dm4")


def combine_datasets(*datasets: VPCFDataset) -> VPCFDataset:
    """
    Combine multiple VPCFDataset objects into one.
    
    Parameters
    ----------
    *datasets : VPCFDataset
        Datasets to combine.
        
    Returns
    -------
    VPCFDataset
        Combined dataset.
    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required.")
    
    # Check feature dimensions match
    dims = [d.feature_dim for d in datasets]
    if len(set(dims)) > 1:
        raise ValueError(f"Feature dimensions do not match: {dims}")
    
    features = np.vstack([d.features for d in datasets])
    
    # Combine labels if all have them
    if all(d.labels is not None for d in datasets):
        labels = np.concatenate([d.labels for d in datasets])
    else:
        labels = None
    
    source_files = ", ".join(d.source_file or "unknown" for d in datasets)
    
    return VPCFDataset(
        features=features,
        raw_images=None,  # Don't combine raw images to save memory
        labels=labels,
        source_file=source_files
    )


# =============================================================================
# Utility Functions
# =============================================================================

def check_dependencies() -> Dict[str, bool]:
    """Check which optional dependencies are available."""
    return {
        "h5py": HAS_H5PY,
        "hyperspy": HAS_HYPERSPY,
        "ncempy": HAS_NCEMPY
    }


def print_file_info(filename: Union[str, Path]) -> None:
    """Print information about a data file."""
    filename = Path(filename)
    ext = filename.suffix.lower()
    
    print(f"File: {filename}")
    print(f"Extension: {ext}")
    print(f"Size: {filename.stat().st_size / (1024*1024):.2f} MB")
    
    if ext in ['.h5', '.hdf5', '.hdf']:
        structure = read_h5_structure(filename)
        print("HDF5 Structure:")
        for group, datasets in structure.items():
            print(f"  {group}/")
            for ds in datasets:
                print(f"    - {ds}")
        print(f"Frame count: {get_h5_frame_count(filename)}")
    
    elif ext in ['.dm3', '.dm4']:
        data, metadata = load_dm3_file(filename)
        print(f"Data shape: {data.shape}")
        print(f"Data dtype: {data.dtype}")


if __name__ == "__main__":
    # Example usage
    deps = check_dependencies()
    print("Available dependencies:")
    for name, available in deps.items():
        status = "[YES]" if available else "[NO]"
        print(f"  {status} {name}")
