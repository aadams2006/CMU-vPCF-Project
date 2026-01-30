"""
Quick-start script for training DEC/IDEC on vPCF_test_2 data files.

This script is pre-configured for:
- C:\Users\alexg\Downloads\vPCF_test_2.h5
- C:\Users\alexg\Downloads\vPCF_test_2.dm3

Usage:
    python run_vpcf_test_2_training.py --model both
    python run_vpcf_test_2_training.py --model dec --n-clusters 5
    python run_vpcf_test_2_training.py --model idec --feature-method histogram
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from train_vpcf_models import run_pipeline


# Pre-configured file paths
H5_FILE = r"C:\Users\alexg\Downloads\vPCF_test_2.h5"
DM3_FILE = r"C:\Users\alexg\Downloads\vPCF_test_2.dm3"

# Output directory
OUTPUT_DIR = r"C:\Users\alexg\Downloads\CMU vPCF Project\Experimentally-obtained vPCF Testing\results"


def train_on_h5_only(
    n_clusters: int = 10,
    feature_method: str = "flatten",
    model: str = "both",
    **kwargs
):
    """
    Train on the HDF5 file only.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    feature_method : str
        Feature extraction method.
    model : str
        Which model: "dec", "idec", or "both".
    **kwargs
        Additional arguments passed to run_pipeline.
    """
    if not os.path.exists(H5_FILE):
        raise FileNotFoundError(f"HDF5 file not found: {H5_FILE}")
    
    print(f"Training on HDF5 file: {H5_FILE}")
    return run_pipeline(
        h5_file=H5_FILE,
        dm3_file=None,
        model=model,
        n_clusters=n_clusters,
        feature_method=feature_method,
        output_dir=os.path.join(OUTPUT_DIR, "h5_only"),
        **kwargs
    )


def train_on_dm3_only(
    n_clusters: int = 10,
    feature_method: str = "flatten",
    model: str = "both",
    **kwargs
):
    """
    Train on the DM3 file only.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    feature_method : str
        Feature extraction method.
    model : str
        Which model: "dec", "idec", or "both".
    **kwargs
        Additional arguments passed to run_pipeline.
    """
    if not os.path.exists(DM3_FILE):
        raise FileNotFoundError(f"DM3 file not found: {DM3_FILE}")
    
    print(f"Training on DM3 file: {DM3_FILE}")
    return run_pipeline(
        h5_file=None,
        dm3_file=DM3_FILE,
        model=model,
        n_clusters=n_clusters,
        feature_method=feature_method,
        output_dir=os.path.join(OUTPUT_DIR, "dm3_only"),
        **kwargs
    )


def train_on_both_files(
    n_clusters: int = 10,
    feature_method: str = "flatten",
    model: str = "both",
    **kwargs
):
    """
    Train on both HDF5 and DM3 files combined.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    feature_method : str
        Feature extraction method.
    model : str
        Which model: "dec", "idec", or "both".
    **kwargs
        Additional arguments passed to run_pipeline.
    """
    files_exist = []
    h5_path = H5_FILE if os.path.exists(H5_FILE) else None
    dm3_path = DM3_FILE if os.path.exists(DM3_FILE) else None
    
    if h5_path is None and dm3_path is None:
        raise FileNotFoundError(
            f"Neither file found:\n  - {H5_FILE}\n  - {DM3_FILE}"
        )
    
    print("Training on combined data from:")
    if h5_path:
        print(f"  - HDF5: {h5_path}")
    if dm3_path:
        print(f"  - DM3: {dm3_path}")
    
    return run_pipeline(
        h5_file=h5_path,
        dm3_file=dm3_path,
        model=model,
        n_clusters=n_clusters,
        feature_method=feature_method,
        output_dir=os.path.join(OUTPUT_DIR, "combined"),
        **kwargs
    )


def main():
    """Interactive main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train DEC/IDEC on vPCF_test_2 data files."
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["h5", "dm3", "both"],
        default="h5",
        help="Which data source(s) to use."
    )
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
        help="Number of clusters."
    )
    parser.add_argument(
        "--feature-method",
        type=str,
        choices=["flatten", "histogram", "statistical", "combined"],
        default="flatten",
        help="Feature extraction method."
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Downsampling factor for images."
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=50,
        help="Pretraining epochs."
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=8000,
        help="Maximum iterations."
    )
    
    args = parser.parse_args()
    
    kwargs = {
        "downsample_factor": args.downsample,
        "pretrain_epochs": args.pretrain_epochs,
        "maxiter": args.maxiter,
    }
    
    if args.source == "h5":
        results = train_on_h5_only(
            n_clusters=args.n_clusters,
            feature_method=args.feature_method,
            model=args.model,
            **kwargs
        )
    elif args.source == "dm3":
        results = train_on_dm3_only(
            n_clusters=args.n_clusters,
            feature_method=args.feature_method,
            model=args.model,
            **kwargs
        )
    else:
        results = train_on_both_files(
            n_clusters=args.n_clusters,
            feature_method=args.feature_method,
            model=args.model,
            **kwargs
        )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    if "dec_labels" in results:
        print(f"DEC clusters: {len(set(results['dec_labels']))} unique")
    if "idec_labels" in results:
        print(f"IDEC clusters: {len(set(results['idec_labels']))} unique")
    
    return results


if __name__ == "__main__":
    main()
