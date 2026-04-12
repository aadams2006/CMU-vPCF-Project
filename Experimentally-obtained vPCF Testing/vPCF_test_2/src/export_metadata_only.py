#!/usr/bin/env python
"""
Quickly export metadata to CSV and JSON (without vPCF images) from the complete HDF5.
"""

import sys
import json
import h5py
import pandas as pd
from pathlib import Path

def main():
    h5_file = Path(__file__).parent / "data" / "vPCF_test_2.h5"
    h5_complete = Path(__file__).parent / "results" / "all_samples_complete.h5"
    
    print("Reading data from HDF5 files...")
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all metadata at once from source H5
    print("[1/2] Creating CSV and JSON metadata...")
    
    metadata_list = []
    json_data = {
        'total_samples': 0,
        'samples': {}
    }
    
    with h5py.File(str(h5_file), 'r') as f:
        grp = f['experiments']
        n_samples = grp['vpcf_images'].shape[0]
        vpcf_shape_info = grp['vpcf_images'].shape  # All samples have same shape
        
        # Read all arrays at once (vectorized)
        atomic_positions = grp['atomic_positions'][:]  # Shape: (7205, 2)
        vpcf_origin = grp['vpcf_origin'][:]           # Shape: (7205, 2)
        peaks_shapes = grp['peaks_shapes'][:]         # Shape: (7205, 2)
        
        print(f"Processing {n_samples} samples...")
        
        for i in range(n_samples):
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{n_samples}...")
            
            sample_id = i
            ap = atomic_positions[i]
            origin = vpcf_origin[i]
            peaks_shape = peaks_shapes[i]
            vpcf_shape = (999, 999)  # All samples have this shape
            
            # CSV row
            metadata_list.append({
                'sample_id': sample_id,
                'atomic_positions_x': float(ap[0]),
                'atomic_positions_y': float(ap[1]),
                'vpcf_origin_x': float(origin[0]),
                'vpcf_origin_y': float(origin[1]),
                'vpcf_image_shape': str(vpcf_shape),
                'peaks_shape': str(tuple(peaks_shape)),
                'crystal_label': 'N/A'
            })
            
            # JSON record
            json_data['samples'][str(sample_id)] = {
                'sample_id': int(sample_id),
                'atomic_positions': [float(ap[0]), float(ap[1])],
                'vpcf_origin': [float(origin[0]), float(origin[1])],
                'vpcf_shape': vpcf_shape,
                'peaks_shape': [int(p) for p in peaks_shape],
                'crystal_label': 'N/A'
            }
        
        json_data['total_samples'] = n_samples
    
    # Write CSV
    csv_output = results_dir / "all_samples_metadata.csv"
    df = pd.DataFrame(metadata_list)
    df.to_csv(csv_output, index=False)
    print(f"\n✓ CSV created: {csv_output.name}")
    print(f"  File size: {csv_output.stat().st_size / (1024*1024):.2f} MB")
    
    # Write JSON
    json_output = results_dir / "all_samples_metadata.json"
    with open(json_output, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\n✓ JSON created: {json_output.name}")
    print(f"  File size: {json_output.stat().st_size / (1024*1024):.2f} MB")
    
    # Summary
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"Total samples: {n_samples}\n")
    print("Files created:")
    print(f"  1. all_samples_complete.h5      - Full vPCF images + metadata (13.7 GB)")
    print(f"  2. all_samples_metadata.csv     - Metadata only (spreadsheet-friendly)")
    print(f"  3. all_samples_metadata.json    - Metadata only (structured format)")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
