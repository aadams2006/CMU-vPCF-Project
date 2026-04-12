"""
H5 Metadata Inspector - Examine all structure, datasets, and attributes in vPCF_test_2.h5

Usage:
    python h5_metadata_inspector.py
    or
    python h5_metadata_inspector.py --sample-indices 0 123 500
    or
    python h5_metadata_inspector.py --check-attributes
"""

from __future__ import annotations

import h5py
import argparse
from pathlib import Path
from typing import Dict, List, Any


def inspect_h5_structure(filename: str | Path) -> Dict[str, Any]:
    """Inspect complete H5 file structure including all groups and datasets."""
    structure = {}
    
    with h5py.File(filename, 'r') as f:
        print("=" * 70)
        print(f"H5 FILE STRUCTURE: {filename}")
        print("=" * 70)
        
        # Root attributes
        print("\n[ROOT ATTRIBUTES]")
        if f.attrs:
            for key, value in f.attrs.items():
                print(f"  {key}: {value} (type: {type(value).__name__})")
        else:
            print("  (none)")
        
        # Root datasets
        print("\n[ROOT DATASETS]")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                ds = f[key]
                print(f"  {key}")
                print(f"    - Shape: {ds.shape}")
                print(f"    - Dtype: {ds.dtype}")
                if ds.attrs:
                    for attr_key, attr_val in ds.attrs.items():
                        print(f"    - Attr '{attr_key}': {attr_val}")
        
        # Root groups
        print("\n[ROOT GROUPS]")
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                group = f[key]
                print(f"  {key}/")
                
                # Group attributes
                if group.attrs:
                    print(f"    [Group Attributes]")
                    for attr_key, attr_val in group.attrs.items():
                        print(f"      {attr_key}: {attr_val}")
                
                # Group datasets
                print(f"    [Datasets]")
                for ds_key in group.keys():
                    if isinstance(group[ds_key], h5py.Dataset):
                        ds = group[ds_key]
                        print(f"      {ds_key}")
                        print(f"        Shape: {ds.shape}, Dtype: {ds.dtype}")
                        
                        # Per-dataset attributes
                        if ds.attrs:
                            for attr_key, attr_val in ds.attrs.items():
                                val_preview = str(attr_val)[:50]
                                print(f"        Attr '{attr_key}': {val_preview}")
                
                # Nested groups
                if any(isinstance(group[k], h5py.Group) for k in group.keys()):
                    print(f"    [Nested Groups]")
                    for sub_key in group.keys():
                        if isinstance(group[sub_key], h5py.Group):
                            print(f"      {sub_key}/")
    
    return structure


def inspect_sample_metadata(filename: str | Path, sample_indices: List[int]) -> None:
    """Inspect metadata for specific samples."""
    with h5py.File(filename, 'r') as f:
        if "experiments" not in f:
            print("ERROR: 'experiments' group not found in H5 file")
            return
        
        grp = f["experiments"]
        n_samples = grp["vpcf_images"].shape[0]
        
        print("\n" + "=" * 70)
        print("SAMPLE-LEVEL METADATA INSPECTION")
        print("=" * 70)
        print(f"Total samples in H5: {n_samples}")
        
        for sample_idx in sample_indices:
            if sample_idx < 0 or sample_idx >= n_samples:
                print(f"\nSample {sample_idx}: OUT OF RANGE [0, {n_samples})")
                continue
            
            print(f"\n[Sample {sample_idx}]")
            
            # Check each dataset for this sample
            for ds_name in grp.keys():
                if isinstance(grp[ds_name], h5py.Dataset):
                    ds = grp[ds_name]
                    if ds.shape[0] > sample_idx:
                        data = ds[sample_idx]
                        
                        # Show summary based on data type
                        if hasattr(data, 'shape'):
                            print(f"  {ds_name}:")
                            print(f"    - Shape: {data.shape if hasattr(data, 'shape') else 'scalar'}")
                            print(f"    - Dtype: {data.dtype if hasattr(data, 'dtype') else type(data).__name__}")
                            if data.size <= 10:
                                print(f"    - Value: {data}")
                            else:
                                print(f"    - Preview: {str(data).split('(')[0]}...")
                        else:
                            print(f"  {ds_name}: {data} (type: {type(data).__name__})")
                        
                        # Check for per-sample attributes
                        if ds.attrs:
                            for attr_key, attr_val in ds.attrs.items():
                                print(f"    - Attr '{attr_key}': {attr_val}")


def check_label_correspondence(h5_file: str | Path, label_file: str | Path = None) -> None:
    """Check if external label file corresponds to H5 samples."""
    print("\n" + "=" * 70)
    print("CHECKING LABEL CORRESPONDENCE")
    print("=" * 70)
    
    with h5py.File(h5_file, 'r') as f:
        h5_samples = f["experiments"]["vpcf_images"].shape[0]
        print(f"H5 file has {h5_samples} samples")
    
    # Check if label file exists in different locations
    search_paths = [
        Path(label_file) if label_file else None,
        Path("data") / "sample_labels.csv",
        Path("data") / "labels.csv",
        Path("..") / "results" / "vpcf_label_mapping.csv",
        Path("results") / "vpcf_label_mapping.csv",
    ]
    
    label_path = None
    for p in search_paths:
        if p and p.exists():
            label_path = p
            break
    
    if label_path:
        print(f"Found label file: {label_path}")
        try:
            import pandas as pd
            df = pd.read_csv(label_path)
            print(f"Label file has {len(df)} entries")
            print(f"Columns: {list(df.columns)}")
            print(f"First 5 rows:")
            print(df.head())
        except Exception as e:
            print(f"Error reading label file: {e}")
    else:
        print("No label file found in expected locations")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect H5 metadata and structure for vPCF_test_2.h5"
    )
    parser.add_argument(
        "--h5-file",
        default="data/vPCF_test_2.h5",
        help="Path to H5 file (default: data/vPCF_test_2.h5)"
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        help="Sample indices to inspect (e.g., 0 123 500)"
    )
    parser.add_argument(
        "--check-attributes",
        action="store_true",
        help="Check for label correspondence with external files"
    )
    
    args = parser.parse_args()
    
    # Inspect main structure
    inspect_h5_structure(args.h5_file)
    
    # Inspect specific samples if requested
    if args.sample_indices:
        inspect_sample_metadata(args.h5_file, args.sample_indices)
    
    # Check for labels
    if args.check_attributes:
        check_label_correspondence(args.h5_file)


if __name__ == "__main__":
    main()
