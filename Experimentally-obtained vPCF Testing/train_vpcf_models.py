"""
Training pipeline for DEC and IDEC models on vPCF data.

This script provides a complete pipeline to:
1. Load vPCF data from HDF5 (.h5) and/or DM3 (.dm3) files
2. Preprocess and extract features
3. Train DEC or IDEC models
4. Save model weights and results

Usage:
    python train_vpcf_models.py --h5-file path/to/file.h5 --model dec
    python train_vpcf_models.py --dm3-file path/to/file.dm3 --model idec
    python train_vpcf_models.py --h5-file file.h5 --dm3-file file.dm3 --model both
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))  # For local imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # For src imports

from vpcf_data_loader import (
    VPCFDataset,
    load_vpcf_file,
    combine_datasets,
    check_dependencies,
    print_file_info,
)
from src.DEC import DEC
from src.IDEC import IDEC


def get_model_dimensions(
    input_dim: int,
    n_clusters: int,
    hidden_dims: Optional[list] = None
) -> list:
    """
    Build the dimension list for the autoencoder.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    n_clusters : int
        Number of clusters (output dimension).
    hidden_dims : list, optional
        Hidden layer dimensions. Defaults to [500, 500, 2000].
        
    Returns
    -------
    list
        Dimension list: [input_dim, hidden1, hidden2, ..., n_clusters]
    """
    if hidden_dims is None:
        hidden_dims = [500, 500, 2000]
    return [input_dim] + hidden_dims + [n_clusters]


def train_dec_model(
    x: np.ndarray,
    y: Optional[np.ndarray],
    n_clusters: int,
    save_dir: str,
    hidden_dims: Optional[list] = None,
    pretrain_epochs: int = 50,
    maxiter: int = 8000,
    update_interval: int = 140,
    batch_size: int = 256,
    verbose: bool = True
) -> Tuple[DEC, np.ndarray]:
    """
    Train a DEC model on the provided features.
    
    Parameters
    ----------
    x : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray, optional
        Ground truth labels for evaluation (can be None).
    n_clusters : int
        Number of clusters to find.
    save_dir : str
        Directory to save model weights and logs.
    hidden_dims : list, optional
        Hidden layer dimensions.
    pretrain_epochs : int
        Number of pretraining epochs.
    maxiter : int
        Maximum clustering iterations.
    update_interval : int
        Interval for updating target distribution.
    batch_size : int
        Batch size for training.
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    tuple
        (trained DEC model, predicted cluster labels)
    """
    dims = get_model_dimensions(x.shape[1], n_clusters, hidden_dims)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Training DEC Model")
        print("=" * 60)
        print(f"Input dimension: {x.shape[1]}")
        print(f"Number of samples: {x.shape[0]}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Network architecture: {dims}")
        print(f"Save directory: {save_dir}")
        print("-" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize DEC
    dec = DEC(dims=dims, n_clusters=n_clusters, save_dir=save_dir)
    
    # Pretrain autoencoder
    if verbose:
        print("\nPhase 1: Pretraining autoencoder...")
    dec.pretrain(x, epochs=pretrain_epochs, batch_size=batch_size)
    
    # Compile and train clustering
    if verbose:
        print("\nPhase 2: Training clustering layer...")
    dec.compile(optimizer='sgd')
    labels = dec.fit(
        x,
        y=y,
        maxiter=maxiter,
        update_interval=update_interval,
        batch_size=batch_size
    )
    
    if verbose:
        print("\nDEC training complete!")
        print(f"Cluster distribution: {np.bincount(labels)}")
    
    return dec, labels


def train_idec_model(
    x: np.ndarray,
    y: Optional[np.ndarray],
    n_clusters: int,
    save_dir: str,
    hidden_dims: Optional[list] = None,
    pretrain_epochs: int = 50,
    maxiter: int = 8000,
    update_interval: int = 140,
    batch_size: int = 256,
    gamma: float = 0.1,
    verbose: bool = True
) -> Tuple[IDEC, np.ndarray]:
    """
    Train an IDEC model on the provided features.
    
    Parameters
    ----------
    x : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray, optional
        Ground truth labels for evaluation (can be None).
    n_clusters : int
        Number of clusters to find.
    save_dir : str
        Directory to save model weights and logs.
    hidden_dims : list, optional
        Hidden layer dimensions.
    pretrain_epochs : int
        Number of pretraining epochs.
    maxiter : int
        Maximum clustering iterations.
    update_interval : int
        Interval for updating target distribution.
    batch_size : int
        Batch size for training.
    gamma : float
        Coefficient for reconstruction loss.
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    tuple
        (trained IDEC model, predicted cluster labels)
    """
    dims = get_model_dimensions(x.shape[1], n_clusters, hidden_dims)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Training IDEC Model")
        print("=" * 60)
        print(f"Input dimension: {x.shape[1]}")
        print(f"Number of samples: {x.shape[0]}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Network architecture: {dims}")
        print(f"Gamma (reconstruction weight): {gamma}")
        print(f"Save directory: {save_dir}")
        print("-" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize IDEC
    idec = IDEC(dims=dims, n_clusters=n_clusters, gamma=gamma, save_dir=save_dir)
    
    # Pretrain autoencoder
    if verbose:
        print("\nPhase 1: Pretraining autoencoder...")
    if not idec.pretrained:
        idec.pretrain(x, epochs=pretrain_epochs, batch_size=batch_size)
    
    # Compile and train clustering
    if verbose:
        print("\nPhase 2: Training clustering layer...")
    idec.compile(optimizer='sgd')
    labels = idec.fit(
        x,
        y=y,
        maxiter=maxiter,
        update_interval=update_interval,
        batch_size=batch_size
    )
    
    if verbose:
        print("\nIDEC training complete!")
        print(f"Cluster distribution: {np.bincount(labels)}")
    
    return idec, labels


def save_results(
    labels: np.ndarray,
    save_dir: str,
    model_name: str,
    source_files: str
) -> None:
    """
    Save clustering results to CSV.
    
    Parameters
    ----------
    labels : np.ndarray
        Predicted cluster labels.
    save_dir : str
        Directory to save results.
    model_name : str
        Name of the model (dec or idec).
    source_files : str
        Source file paths for reference.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save labels
    labels_df = pd.DataFrame({
        'sample_idx': np.arange(len(labels)),
        'cluster': labels
    })
    labels_path = os.path.join(save_dir, f'{model_name}_cluster_labels.csv')
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved cluster labels to: {labels_path}")
    
    # Save cluster statistics
    unique, counts = np.unique(labels, return_counts=True)
    stats_df = pd.DataFrame({
        'cluster': unique,
        'count': counts,
        'percentage': counts / len(labels) * 100
    })
    stats_path = os.path.join(save_dir, f'{model_name}_cluster_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved cluster statistics to: {stats_path}")
    
    # Save metadata
    with open(os.path.join(save_dir, f'{model_name}_metadata.txt'), 'w') as f:
        f.write(f"Source files: {source_files}\n")
        f.write(f"Total samples: {len(labels)}\n")
        f.write(f"Number of clusters: {len(unique)}\n")


def run_pipeline(
    h5_file: Optional[str] = None,
    dm3_file: Optional[str] = None,
    model: str = "both",
    n_clusters: int = 10,
    feature_method: str = "flatten",
    normalize: str = "minmax",
    downsample_factor: Optional[int] = None,
    hidden_dims: Optional[list] = None,
    pretrain_epochs: int = 50,
    maxiter: int = 8000,
    update_interval: int = 140,
    batch_size: int = 256,
    output_dir: str = "results/vpcf",
    verbose: bool = True
) -> dict:
    """
    Run the complete training pipeline.
    
    Parameters
    ----------
    h5_file : str, optional
        Path to HDF5 file.
    dm3_file : str, optional
        Path to DM3 file.
    model : str
        Which model to train: "dec", "idec", or "both".
    n_clusters : int
        Number of clusters.
    feature_method : str
        Feature extraction method: "flatten", "histogram", "statistical", or "combined"
    normalize : str
        Normalization method.
    downsample_factor : int, optional
        Downsampling factor for images.
    hidden_dims : list, optional
        Hidden layer dimensions.
    pretrain_epochs : int
        Pretraining epochs.
    maxiter : int
        Maximum iterations.
    update_interval : int
        Update interval.
    batch_size : int
        Batch size.
    output_dir : str
        Output directory.
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    dict
        Results containing models and labels.
    """
    if h5_file is None and dm3_file is None:
        raise ValueError("At least one of h5_file or dm3_file must be provided.")
    
    # Check dependencies
    deps = check_dependencies()
    if verbose:
        print("Checking dependencies:")
        for name, available in deps.items():
            status = "✓" if available else "✗"
            print(f"  {status} {name}")
        print()
    
    # Load datasets
    datasets = []
    
    if h5_file is not None:
        if verbose:
            print(f"Loading HDF5 file: {h5_file}")
            print_file_info(h5_file)
            print()
        
        h5_dataset = load_vpcf_file(
            h5_file,
            feature_method=feature_method,
            normalize=normalize,
            downsample_factor=downsample_factor,
            verbose=verbose
        )
        datasets.append(h5_dataset)
        if verbose:
            print(f"Loaded: {h5_dataset}")
            print()
    
    if dm3_file is not None:
        if verbose:
            print(f"Loading DM3 file: {dm3_file}")
            print_file_info(dm3_file)
            print()
        
        dm3_dataset = load_vpcf_file(
            dm3_file,
            feature_method=feature_method,
            normalize=normalize,
            downsample_factor=downsample_factor,
            verbose=verbose
        )
        datasets.append(dm3_dataset)
        if verbose:
            print(f"Loaded: {dm3_dataset}")
            print()
    
    # Combine datasets if multiple
    if len(datasets) > 1:
        if verbose:
            print("Combining datasets...")
        dataset = combine_datasets(*datasets)
    else:
        dataset = datasets[0]
    
    if verbose:
        print(f"\nFinal dataset: {dataset}")
        print(f"Feature matrix shape: {dataset.features.shape}")
        print()
    
    # Get features
    x = dataset.features
    y = dataset.labels  # May be None
    
    results = {
        "dataset": dataset,
        "features": x,
        "labels_true": y
    }
    
    # Train models
    if model in ["dec", "both"]:
        dec_save_dir = os.path.join(output_dir, "dec")
        dec_model, dec_labels = train_dec_model(
            x=x,
            y=y,
            n_clusters=n_clusters,
            save_dir=dec_save_dir,
            hidden_dims=hidden_dims,
            pretrain_epochs=pretrain_epochs,
            maxiter=maxiter,
            update_interval=update_interval,
            batch_size=batch_size,
            verbose=verbose
        )
        results["dec_model"] = dec_model
        results["dec_labels"] = dec_labels
        save_results(dec_labels, dec_save_dir, "dec", dataset.source_file or "")
    
    if model in ["idec", "both"]:
        idec_save_dir = os.path.join(output_dir, "idec")
        idec_model, idec_labels = train_idec_model(
            x=x,
            y=y,
            n_clusters=n_clusters,
            save_dir=idec_save_dir,
            hidden_dims=hidden_dims,
            pretrain_epochs=pretrain_epochs,
            maxiter=maxiter,
            update_interval=update_interval,
            batch_size=batch_size,
            verbose=verbose
        )
        results["idec_model"] = idec_model
        results["idec_labels"] = idec_labels
        save_results(idec_labels, idec_save_dir, "idec", dataset.source_file or "")
    
    if verbose:
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)
        print(f"Results saved to: {output_dir}")
    
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train DEC/IDEC models on vPCF data from HDF5 and/or DM3 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument(
        "--h5-file",
        type=str,
        default=None,
        help="Path to HDF5 (.h5) file containing vPCF data."
    )
    parser.add_argument(
        "--dm3-file",
        type=str,
        default=None,
        help="Path to DM3 file containing microscopy data."
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["dec", "idec", "both"],
        default="both",
        help="Which model(s) to train."
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters to find."
    )
    
    # Feature extraction
    parser.add_argument(
        "--feature-method",
        type=str,
        choices=["flatten", "histogram", "statistical", "combined"],
        default="flatten",
        help="Feature extraction method."
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["minmax", "standard", "l2", "none"],
        default="minmax",
        help="Normalization method (use 'none' to skip)."
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Downsampling factor for images."
    )
    
    # Model architecture
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[500, 500, 2000],
        help="Hidden layer dimensions for autoencoder."
    )
    
    # Training parameters
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=50,
        help="Number of pretraining epochs."
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=8000,
        help="Maximum clustering iterations."
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=140,
        help="Interval for updating target distribution."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training."
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/vpcf",
        help="Directory to save results."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output."
    )
    
    return parser


def main():
    """Main entry point for command-line usage."""
    parser = build_argument_parser()
    args = parser.parse_args()
    
    if args.h5_file is None and args.dm3_file is None:
        parser.error("At least one of --h5-file or --dm3-file must be provided.")
    
    normalize = args.normalize if args.normalize != "none" else None
    
    run_pipeline(
        h5_file=args.h5_file,
        dm3_file=args.dm3_file,
        model=args.model,
        n_clusters=args.n_clusters,
        feature_method=args.feature_method,
        normalize=normalize,
        downsample_factor=args.downsample,
        hidden_dims=args.hidden_dims,
        pretrain_epochs=args.pretrain_epochs,
        maxiter=args.maxiter,
        update_interval=args.update_interval,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
