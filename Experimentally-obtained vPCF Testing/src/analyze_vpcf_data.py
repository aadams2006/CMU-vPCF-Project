#!/usr/bin/env python
"""
Detailed analysis of vPCF origin and peak data interpretation.
"""

import h5py
import numpy as np
from pathlib import Path

h5_file = Path("data/vPCF_test_2.h5")

print("="*80)
print("vPCF ORIGIN AND PEAK DATA ANALYSIS")
print("="*80)

with h5py.File(h5_file, 'r') as f:
    grp = f['experiments']
    
    # === vPCF Origin ===
    print("\n[1] vPCF ORIGIN - What does it mean?")
    print("-"*80)
    
    origins = grp['vpcf_origin'][:]
    print(f"Shape: {origins.shape}")
    print(f"Data type: {origins.dtype}")
    print(f"\nFirst 10 samples:")
    for i in range(10):
        print(f"  Sample {i}: origin = {origins[i]}")
    
    print(f"\nUnique values check:")
    unique_origins = np.unique(origins, axis=0)
    print(f"  Number of unique origins: {len(unique_origins)}")
    print(f"  Unique origins: {unique_origins}")
    
    print("\nInterpretation:")
    print("  - vPCF origin represents the [X, Y] reference point in the 999×999 image")
    print("  - For HfO data, this is typically [500, 498]")
    print("  - All samples appear to use the SAME origin (centered at ~center of image)")
    print("  - This is the coordinate system baseline for the vPCF spatial analysis")
    
    # === Peak Data ===
    print("\n" + "="*80)
    print("[2] PEAK DATA - What does it contain?")
    print("-"*80)
    
    peaks_shapes = grp['peaks_shapes'][:]
    vpcf_peaks = grp['vpcf_peaks']
    
    print(f"Peaks shapes - Shape: {peaks_shapes.shape}")
    print(f"Peaks shapes - Data type: {peaks_shapes.dtype}")
    print(f"\nFirst 10 samples peak shapes:")
    for i in range(10):
        print(f"  Sample {i}: peak_shape = {peaks_shapes[i]}")
    
    print(f"\nUnique peak shapes:")
    unique_peaks = np.unique(peaks_shapes, axis=0)
    for u in unique_peaks:
        count = np.sum((peaks_shapes == u).all(axis=1))
        print(f"  Shape {u}: found in {count} samples")
    
    print("\n\nDetailed peak data (vpcf_peaks object array):")
    print(f"Type: {vpcf_peaks.dtype}")
    print(f"Shape: {vpcf_peaks.shape}")
    
    # Look at actual peak data for a few samples
    print(f"\nSample 0 peaks (first 5 rows):")
    peaks_0 = vpcf_peaks[0]
    print(f"  Type: {type(peaks_0)}")
    print(f"  Shape: {peaks_0.shape if hasattr(peaks_0, 'shape') else 'N/A'}")
    print(f"  Data:\n{peaks_0[:5] if hasattr(peaks_0, '__getitem__') else peaks_0}")
    
    print(f"\nSample 123 peaks (first 5 rows):")
    peaks_123 = vpcf_peaks[123]
    print(f"  Type: {type(peaks_123)}")
    print(f"  Shape: {peaks_123.shape if hasattr(peaks_123, 'shape') else 'N/A'}")
    print(f"  Data:\n{peaks_123[:5] if hasattr(peaks_123, '__getitem__') else peaks_123}")
    
    print("\nInterpretation:")
    print("  - peaks_shape = [num_peaks, num_attributes_per_peak]")
    print("  - For HfO, typically [76, 5] meaning 76 detected peaks, 5 attributes each")
    print("  - vpcf_peaks contains the actual peak data (coordinates, intensities, etc.)")
    print("  - Peak attributes likely: [x, y, intensity, width, height] or similar")
    
    # === Crystal Type / Labels ===
    print("\n" + "="*80)
    print("[3] CRYSTAL TYPE - Why is it not available?")
    print("-"*80)
    
    print("\nH5 file structure inspection:")
    print(f"Root groups: {list(f.keys())}")
    print(f"Experiments group datasets: {list(grp.keys())}")
    print(f"Experiments group attributes: {dict(grp.attrs)}")
    
    print("\nConclusion:")
    print("  [NOT IN H5] Crystal type/structure information is NOT stored in the H5 file")
    print("  [ONLY IN H5] The H5 file contains ONLY raw experimental vPCF data:")
    print("    - vPCF images (2D spatial correlation functions)")
    print("    - Atomic positions")
    print("    - Peak data")
    print("  [AVAILABLE] Crystal labels ARE available in external files:")
    print(f"    - {Path(__file__).parent / 'results' / 'vpcf_label_mapping.csv'}")
    print("  ")
    print("  This separation is good design - it keeps:")
    print("  - Measurement data (H5) separate from")
    print("  - Classification data (CSV labels)")

print("\n" + "="*80)
