"""
Sample Metadata Lookup - Unified interface for querying vPCF samples by metadata.

Provides batch queries, crystal structure type searches, and metadata export.

Usage:
    from sample_metadata_lookup import SampleMetadataManager
    
    manager = SampleMetadataManager("data/vPCF_test_2.h5")
    
    # Get complete info for sample 123
    info = manager.get_sample_info(123)
    
    # Search by crystal type
    tetra_samples = manager.search_by_crystal_type("Tetragonal")
    
    # Export metadata for cluster
    manager.export_sample_metadata(
        sample_ids=[0, 15, 42, 100],
        output_file="cluster_0_metadata.csv"
    )
"""

from __future__ import annotations

import h5py
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import numpy as np


class SampleMetadataManager:
    """Unified metadata access for vPCF samples."""
    
    def __init__(
        self,
        h5_filepath: str | Path,
        external_labels_csv: Optional[str | Path] = None,
        verbose: bool = True
    ):
        """
        Initialize the metadata manager.
        
        Parameters
        ----------
        h5_filepath : str or Path
            Path to vPCF_test_2.h5
        external_labels_csv : str or Path, optional
            Path to CSV with crystal structure labels (e.g., vpcf_label_mapping.csv)
            If None, will search common locations.
        verbose : bool
            Whether to print debug info
        """
        self.h5_filepath = Path(h5_filepath)
        self.verbose = verbose
        
        if not self.h5_filepath.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_filepath}")
        
        # Load H5 metadata
        self._load_h5_info()
        
        # Load external labels if available
        self.labels_df = None
        self._load_labels(external_labels_csv)
    
    def _load_h5_info(self) -> None:
        """Load basic H5 structure info."""
        with h5py.File(self.h5_filepath, 'r') as f:
            grp = f["experiments"]
            self.n_samples = grp["vpcf_images"].shape[0]
            self.has_atomic_positions = "atomic_positions" in grp
            self.has_vpcf_origin = "vpcf_origin" in grp
            self.has_peaks = "vpcf_peaks" in grp
            self.has_peaks_shapes = "peaks_shapes" in grp
        
        if self.verbose:
            print(f"Loaded H5 file: {self.n_samples} samples")
    
    def _load_labels(self, labels_csv: Optional[str | Path] = None) -> None:
        """Try to load external label file."""
        # Explicit path provided
        if labels_csv:
            label_path = Path(labels_csv)
            if label_path.exists():
                try:
                    self.labels_df = pd.read_csv(label_path)
                    if self.verbose:
                        print(f"Loaded labels from: {label_path}")
                    return
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading labels from {label_path}: {e}")
        
        # Search common locations
        search_paths = [
            self.h5_filepath.parent / "sample_labels.csv",
            self.h5_filepath.parent / "labels.csv",
            self.h5_filepath.parent.parent / "results" / "vpcf_label_mapping.csv",
            Path("results") / "vpcf_label_mapping.csv",
        ]
        
        for path in search_paths:
            if path.exists():
                try:
                    self.labels_df = pd.read_csv(path)
                    if self.verbose:
                        print(f"Found labels at: {path}")
                    return
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading {path}: {e}")
        
        if self.verbose:
            print("No external label file found")
    
    def get_sample_info(self, sample_id: int, include_image: bool = False) -> Dict[str, Any]:
        """
        Get complete information for a single sample.
        
        Parameters
        ----------
        sample_id : int
            Sample index (0-based)
        include_image : bool
            Whether to include vPCF image array
            
        Returns
        -------
        dict
            Complete sample information including crystal type if available
        """
        if sample_id < 0 or sample_id >= self.n_samples:
            raise IndexError(f"Sample {sample_id} out of range [0, {self.n_samples})")
        
        info = {'sample_id': sample_id}
        
        # H5 data
        with h5py.File(self.h5_filepath, 'r') as f:
            grp = f["experiments"]
            
            if self.has_atomic_positions:
                info['atomic_positions'] = grp["atomic_positions"][sample_id][:]
            
            if self.has_vpcf_origin:
                info['vpcf_origin'] = grp["vpcf_origin"][sample_id][:]
            
            info['vpcf_shape'] = grp["vpcf_images"][sample_id].shape
            
            if include_image:
                info['vpcf_image'] = grp["vpcf_images"][sample_id][:]
            
            if self.has_peaks_shapes:
                info['peaks_shape'] = tuple(grp["peaks_shapes"][sample_id])
        
        # External labels if available
        if self.labels_df is not None:
            # Try to match by index
            if 'encoded' in self.labels_df.columns and sample_id < len(self.labels_df):
                label_row = self.labels_df.iloc[sample_id]
                if 'label' in label_row:
                    info['crystal_label'] = label_row['label']
                    
                    # Parse crystal type and orientation from label
                    label_str = label_row['label']
                    parts = label_str.split('_')
                    if len(parts) > 0:
                        info['crystal_type'] = parts[0]
                        if len(parts) > 1:
                            info['orientation_vector'] = parts[1]
        
        return info
    
    def search_by_crystal_type(self, crystal_type: str) -> List[int]:
        """
        Find all samples with a specific crystal type.
        
        Parameters
        ----------
        crystal_type : str
            Crystal type (e.g., "Tetragonal", "Monoclinic", "Ortho-Pbca")
            
        Returns
        -------
        list
            Sample indices with that crystal type
        """
        if self.labels_df is None:
            raise ValueError("No label data available. Cannot search by crystal type.")
        
        matching_samples = []
        
        for idx in range(len(self.labels_df)):
            if idx < self.n_samples:
                label = self.labels_df.iloc[idx]['label']
                if label.startswith(crystal_type):
                    matching_samples.append(idx)
        
        return matching_samples
    
    def get_crystal_types(self) -> Dict[str, List[int]]:
        """
        Get all crystal types and their associated sample indices.
        
        Returns
        -------
        dict
            {crystal_type: [sample_ids]}
        """
        if self.labels_df is None:
            raise ValueError("No label data available.")
        
        crystal_types = {}
        
        for idx in range(len(self.labels_df)):
            if idx < self.n_samples:
                label = self.labels_df.iloc[idx]['label']
                crystal_type = label.split('_')[0]
                
                if crystal_type not in crystal_types:
                    crystal_types[crystal_type] = []
                crystal_types[crystal_type].append(idx)
        
        return crystal_types
    
    def export_sample_metadata(
        self,
        sample_ids: List[int],
        output_file: str | Path,
        include_images: bool = False
    ) -> Dict[str, Any]:
        """
        Export metadata for multiple samples to CSV.
        
        Parameters
        ----------
        sample_ids : list
            List of sample indices to export
        output_file : str or Path
            Output CSV file path
        include_images : bool
            Whether to save images (creates separate folder)
            
        Returns
        -------
        dict
            Export summary with file locations
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for sample_id in sample_ids:
            try:
                info = self.get_sample_info(sample_id, include_image=False)
                
                row = {
                    'sample_id': sample_id,
                    'atomic_x': info['atomic_positions'][0] if 'atomic_positions' in info else None,
                    'atomic_y': info['atomic_positions'][1] if 'atomic_positions' in info else None,
                    'vpcf_origin_x': info['vpcf_origin'][0] if 'vpcf_origin' in info else None,
                    'vpcf_origin_y': info['vpcf_origin'][1] if 'vpcf_origin' in info else None,
                    'vpcf_shape': str(info['vpcf_shape']),
                }
                
                if 'crystal_label' in info:
                    row['crystal_label'] = info['crystal_label']
                    row['crystal_type'] = info.get('crystal_type')
                    row['orientation_vector'] = info.get('orientation_vector')
                
                rows.append(row)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing sample {sample_id}: {e}")
        
        # Save CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            
            summary = {
                'output_file': str(output_file),
                'samples_exported': len(rows),
                'total_requested': len(sample_ids),
                'columns': list(df.columns)
            }
            
            if self.verbose:
                print(f"Exported {len(rows)} samples to: {output_file}")
            
            return summary
        else:
            return {'error': 'No samples exported'}
    
    def create_summary_report(self) -> Dict[str, Any]:
        """Create a summary report of available metadata."""
        report = {
            'total_samples': self.n_samples,
            'has_atomic_positions': self.has_atomic_positions,
            'has_vpcf_origin': self.has_vpcf_origin,
            'has_peaks': self.has_peaks,
            'has_peaks_shapes': self.has_peaks_shapes,
            'has_crystal_labels': self.labels_df is not None,
        }
        
        if self.labels_df is not None:
            crystal_types = self.get_crystal_types()
            report['crystal_types'] = {
                ct: len(samples) for ct, samples in crystal_types.items()
            }
            report['num_crystal_types'] = len(crystal_types)
        
        return report
    
    def print_summary(self) -> None:
        """Print a readable summary of metadata availability."""
        report = self.create_summary_report()
        
        print("\n" + "=" * 70)
        print("SAMPLE METADATA SUMMARY")
        print("=" * 70)
        print(f"Total samples: {report['total_samples']}")
        print(f"Atomic positions: {'✓' if report['has_atomic_positions'] else '✗'}")
        print(f"vPCF origin: {'✓' if report['has_vpcf_origin'] else '✗'}")
        print(f"Peak data: {'✓' if report['has_peaks'] else '✗'}")
        print(f"Crystal labels: {'✓' if report['has_crystal_labels'] else '✗'}")
        
        if report['has_crystal_labels']:
            print(f"\nCrystal types ({report['num_crystal_types']}):")
            for ct, count in sorted(report['crystal_types'].items()):
                print(f"  {ct}: {count} samples")
