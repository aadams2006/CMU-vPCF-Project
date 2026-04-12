#!/usr/bin/env python
"""
Export all sample IDs with their complete vPCF information including images.
Exports to both HDF5 (with full data) and CSV (metadata only).
"""

import sys
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sample_metadata_lookup import SampleMetadataManager

def main():
    # Get the data directory path
    data_dir = Path(__file__).parent / "data"
    h5_file = data_dir / "vPCF_test_2.h5"
    
    if not h5_file.exists():
        print(f"Error: H5 file not found at {h5_file}")
        return 1
    
    print("Initializing SampleMetadataManager...")
    manager = SampleMetadataManager(str(h5_file))
    
    # Get all sample IDs (0 to 7204)
    all_sample_ids = list(range(manager.n_samples))
    
    print(f"Exporting complete data for all {len(all_sample_ids)} samples...")
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # 1. Export to HDF5 with full data (vPCF images + metadata)
    # ============================================================================
    h5_output = results_dir / "all_samples_complete.h5"
    print(f"\n[1/3] Creating HDF5 file with vPCF images and metadata...")
    
    with h5py.File(h5_output, 'w') as out_f:
        # Create group for samples
        samples_grp = out_f.create_group('samples')
        
        # Open source H5 file to read vPCF images
        with h5py.File(h5_file, 'r') as in_f:
            in_grp = in_f['experiments']
            
            # Create datasets for each sample
            for i, sample_id in enumerate(all_sample_ids):
                if (i + 1) % 1000 == 0:
                    print(f"  Processing sample {i + 1}/{len(all_sample_ids)}...")
                
                # Get all info including the vPCF image
                info = manager.get_sample_info(sample_id, include_image=True)
                
                # Create a group for this sample
                sample_grp = samples_grp.create_group(f"sample_{sample_id:05d}")
                
                # Store all metadata
                sample_grp.attrs['sample_id'] = sample_id
                sample_grp.attrs['atomic_positions_x'] = float(info['atomic_positions'][0])
                sample_grp.attrs['atomic_positions_y'] = float(info['atomic_positions'][1])
                sample_grp.attrs['vpcf_origin_x'] = float(info['vpcf_origin'][0])
                sample_grp.attrs['vpcf_origin_y'] = float(info['vpcf_origin'][1])
                sample_grp.attrs['vpcf_shape'] = str(info['vpcf_shape'])
                sample_grp.attrs['peaks_shape'] = str(info['peaks_shape'])
                
                if 'crystal_label' in info and info['crystal_label'] is not None:
                    sample_grp.attrs['crystal_label'] = str(info['crystal_label'])
                
                # Store the vPCF image as a dataset
                sample_grp.create_dataset('vpcf_image', data=info['vpcf_image'], dtype='float16')
        
        print(f"✓ HDF5 file created: {h5_output}")
        print(f"  File size: {h5_output.stat().st_size / (1024*1024):.2f} MB")
    
    # ============================================================================
    # 2. Export metadata only to CSV (for easy viewing in spreadsheets)
    # ============================================================================
    csv_output = results_dir / "all_samples_metadata.csv"
    print(f"\n[2/3] Creating CSV with metadata...")
    
    metadata_list = []
    for i, sample_id in enumerate(all_sample_ids):
        if (i + 1) % 1000 == 0:
            print(f"  Processing sample {i + 1}/{len(all_sample_ids)}...")
        
        info = manager.get_sample_info(sample_id, include_image=False)
        metadata_list.append({
            'sample_id': sample_id,
            'atomic_positions_x': info['atomic_positions'][0],
            'atomic_positions_y': info['atomic_positions'][1],
            'vpcf_origin_x': info['vpcf_origin'][0],
            'vpcf_origin_y': info['vpcf_origin'][1],
            'vpcf_image_shape': str(info['vpcf_shape']),
            'peaks_shape': str(info['peaks_shape']),
            'crystal_label': info.get('crystal_label', 'N/A')
        })
    
    df = pd.DataFrame(metadata_list)
    df.to_csv(csv_output, index=False)
    print(f"✓ CSV file created: {csv_output}")
    print(f"  File size: {csv_output.stat().st_size / (1024*1024):.2f} MB")
    
    # ============================================================================
    # 3. Export to JSON (alternative structured format)
    # ============================================================================
    json_output = results_dir / "all_samples_metadata.json"
    print(f"\n[3/3] Creating JSON with complete metadata...")
    
    json_data = {
        'total_samples': len(all_sample_ids),
        'samples': {}
    }
    
    for i, sample_id in enumerate(all_sample_ids):
        if (i + 1) % 1000 == 0:
            print(f"  Processing sample {i + 1}/{len(all_sample_ids)}...")
        
        info = manager.get_sample_info(sample_id, include_image=False)
        json_data['samples'][str(sample_id)] = {
            'sample_id': int(sample_id),
            'atomic_positions': [float(info['atomic_positions'][0]), float(info['atomic_positions'][1])],
            'vpcf_origin': [float(info['vpcf_origin'][0]), float(info['vpcf_origin'][1])],
            'vpcf_shape': info['vpcf_shape'],
            'peaks_shape': [int(x) for x in info['peaks_shape']],
            'crystal_label': str(info.get('crystal_label', 'N/A'))
        }
    
    with open(json_output, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✓ JSON file created: {json_output}")
    print(f"  File size: {json_output.stat().st_size / (1024*1024):.2f} MB")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPORT SUMMARY")
    print("="*70)
    print(f"Total samples exported: {len(all_sample_ids)}")
    print("\nFiles created:")
    print(f"  1. {h5_output.name}")
    print(f"     - Contains full vPCF images (999x999) + metadata for all samples")
    print(f"  2. {csv_output.name}")
    print(f"     - Metadata only (viewable in Excel/spreadsheet)")
    print(f"  3. {json_output.name}")
    print(f"     - Structured metadata in JSON format")
    print("\nDataset Summary:")
    manager.print_summary()
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
