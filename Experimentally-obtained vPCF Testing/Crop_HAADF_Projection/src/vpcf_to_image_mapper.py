"""
Map vPCF cluster assignments and features back to original image coordinates.

This module provides functionality to:
- Link vPCF samples to their original image locations
- Retrieve cluster assignments with coordinates
- Query samples by cluster or coordinate region
- Export mapping data for visualization
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import h5py


class VPCFToImageMapper:
    """Maps vPCF clustering results back to original image coordinates."""
    
    def __init__(self, 
                 h5_file: str,
                 clustering_results_dir: str,
                 method: str = 'dec'):
        """
        Initialize the mapper.
        
        Parameters
        ----------
        h5_file : str
            Path to the H5 data file (e.g., 1730_poled_TaN_cropped_100.h5)
        clustering_results_dir : str
            Path to clustering results directory (e.g., h5_only/dec/)
        method : str, optional
            Clustering method: 'dec' or 'idec' (default: 'dec')
        """
        self.h5_file = Path(h5_file)
        self.clustering_results_dir = Path(clustering_results_dir)
        self.method = method
        
        # Load metadata from H5 file
        self._load_metadata()
        
        # Load cluster assignments
        self._load_cluster_assignments()
    
    def _load_metadata(self):
        """Load metadata from H5 file."""
        if not self.h5_file.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_file}")
        
        with h5py.File(self.h5_file, 'r') as f:
            exp = f['experiments']
            
            # Get dimensions
            self.total_samples = len(exp['atomic_positions'])
            
            # Load metadata into dictionary
            self.metadata = {}
            
            atomic_positions = exp['atomic_positions'][:]
            vpcf_origins = exp['vpcf_origin'][:]
            peaks_shapes = exp['peaks_shapes'][:]
            
            for i in range(self.total_samples):
                self.metadata[str(i)] = {
                    'sample_id': i,
                    'atomic_positions': [float(atomic_positions[i, 0]), float(atomic_positions[i, 1])],
                    'vpcf_origin': [float(vpcf_origins[i, 0]), float(vpcf_origins[i, 1])],
                    'vpcf_shape': (999, 999),  # From the H5 file structure
                    'peaks_shape': tuple(peaks_shapes[i]),
                    'crystal_label': 'N/A'  # Not available in H5 file
                }
        
        print(f"Loaded metadata for {self.total_samples} samples from {self.h5_file.name}")
    
    def _load_cluster_assignments(self):
        """Load cluster assignments from CSV file."""
        cluster_file = self.clustering_results_dir / f'{self.method}_cluster_labels.csv'
        
        if not cluster_file.exists():
            raise FileNotFoundError(f"Cluster labels file not found: {cluster_file}")
        
        df = pd.read_csv(cluster_file)
        # Map sample_idx to cluster
        self.clusters = dict(zip(df['sample_idx'], df['cluster']))
        self.n_clusters = max(self.clusters.values()) + 1
        
        print(f"Loaded {len(self.clusters)} samples with {self.n_clusters} clusters")
    
    def get_sample_info(self, sample_id: int) -> Dict:
        """
        Get complete information for a sample including cluster and image coordinates.
        
        Parameters
        ----------
        sample_id : int
            Sample index
        
        Returns
        -------
        dict
            Dictionary containing:
            - cluster: cluster assignment
            - atomic_x, atomic_y: coordinates in original image
            - vpcf_origin_x, vpcf_origin_y: vPCF window origin
            - vpcf_shape: vPCF image dimensions
            - metadata: full metadata entry
        """
        if sample_id not in self.clusters:
            raise ValueError(f"Sample {sample_id} not found in cluster assignments")
        
        meta = self.metadata[str(sample_id)]
        
        return {
            'sample_id': sample_id,
            'cluster': self.clusters[sample_id],
            'atomic_x': meta['atomic_positions'][0],
            'atomic_y': meta['atomic_positions'][1],
            'vpcf_origin_x': meta['vpcf_origin'][0],
            'vpcf_origin_y': meta['vpcf_origin'][1],
            'vpcf_shape': meta['vpcf_shape'],
            'peaks_shape': meta['peaks_shape'],
            'crystal_label': meta.get('crystal_label', 'N/A'),
            'metadata': meta
        }
    
    def get_cluster_samples(self, cluster_id: int) -> pd.DataFrame:
        """
        Get all samples in a cluster with their image coordinates.
        
        Parameters
        ----------
        cluster_id : int
            Cluster index
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: sample_id, cluster, atomic_x, atomic_y, 
            vpcf_origin_x, vpcf_origin_y, vpcf_shape
        """
        sample_ids = [s for s, c in self.clusters.items() if c == cluster_id]
        
        results = []
        for sid in sample_ids:
            info = self.get_sample_info(sid)
            results.append({
                'sample_id': info['sample_id'],
                'cluster': info['cluster'],
                'atomic_x': info['atomic_x'],
                'atomic_y': info['atomic_y'],
                'vpcf_origin_x': info['vpcf_origin_x'],
                'vpcf_origin_y': info['vpcf_origin_y'],
                'vpcf_shape_str': str(info['vpcf_shape']),
            })
        
        return pd.DataFrame(results)
    
    def get_samples_in_region(self, 
                              x_min: float, x_max: float,
                              y_min: float, y_max: float) -> pd.DataFrame:
        """
        Get all samples with atomic positions within a region of the original image.
        
        Parameters
        ----------
        x_min, x_max, y_min, y_max : float
            Region boundaries in image coordinates
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: sample_id, cluster, atomic_x, atomic_y, etc.
        """
        results = []
        for sample_id in range(self.total_samples):
            if sample_id in self.clusters:
                info = self.get_sample_info(sample_id)
                ax, ay = info['atomic_x'], info['atomic_y']
                
                if x_min <= ax <= x_max and y_min <= ay <= y_max:
                    results.append({
                        'sample_id': info['sample_id'],
                        'cluster': info['cluster'],
                        'atomic_x': info['atomic_x'],
                        'atomic_y': info['atomic_y'],
                        'vpcf_origin_x': info['vpcf_origin_x'],
                        'vpcf_origin_y': info['vpcf_origin_y'],
                        'vpcf_shape_str': str(info['vpcf_shape']),
                    })
        
        return pd.DataFrame(results)
    
    def export_cluster_coordinates(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Export all samples with cluster assignments and coordinates.
        
        Parameters
        ----------
        output_file : str, optional
            If provided, save to CSV file
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all samples and their info
        """
        results = []
        for sample_id in range(self.total_samples):
            if sample_id in self.clusters:
                info = self.get_sample_info(sample_id)
                results.append({
                    'sample_id': info['sample_id'],
                    'cluster': info['cluster'],
                    'atomic_x': info['atomic_x'],
                    'atomic_y': info['atomic_y'],
                    'vpcf_origin_x': info['vpcf_origin_x'],
                    'vpcf_origin_y': info['vpcf_origin_y'],
                    'vpcf_shape': info['vpcf_shape'][0],  # assuming square
                    'crystal_label': info['crystal_label'],
                })
        
        df = pd.DataFrame(results)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Exported to: {output_file}")
        
        return df
    
    def get_cluster_statistics(self) -> Dict:
        """
        Get statistics about cluster distribution and spatial distribution.
        
        Returns
        -------
        dict
            Statistics including cluster sizes, coordinate ranges, etc.
        """
        stats = {}
        
        for cluster_id in range(self.n_clusters):
            df = self.get_cluster_samples(cluster_id)
            
            if len(df) > 0:
                stats[f'cluster_{cluster_id}'] = {
                    'n_samples': len(df),
                    'atomic_x_mean': df['atomic_x'].mean(),
                    'atomic_y_mean': df['atomic_y'].mean(),
                    'atomic_x_std': df['atomic_x'].std(),
                    'atomic_y_std': df['atomic_y'].std(),
                    'atomic_x_min': df['atomic_x'].min(),
                    'atomic_x_max': df['atomic_x'].max(),
                    'atomic_y_min': df['atomic_y'].min(),
                    'atomic_y_max': df['atomic_y'].max(),
                }
        
        return stats
    
    def summary(self):
        """Print summary of loaded data."""
        print(f"\n{'='*60}")
        print(f"vPCF to Image Mapper Summary")
        print(f"{'='*60}")
        print(f"Total samples: {self.total_samples}")
        print(f"Samples with clusters: {len(self.clusters)}")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Clustering method: {self.method}")
        print(f"{'='*60}\n")


# Example usage demonstrating how to use the mapper
if __name__ == "__main__":
    import sys
    
    # Configure paths based on your workspace structure
    workspace_root = Path(__file__).parent.parent
    h5_file = workspace_root / "data" / "Crop_HAADF_Projection_80pixels.h5"
    clustering_dir = workspace_root / "results" / "h5_only" / "dec"
    
    # Initialize mapper
    mapper = VPCFToImageMapper(
        h5_file=str(h5_file),
        clustering_results_dir=str(clustering_dir),
        method='dec'
    )
    
    mapper.summary()
    
    # Example 1: Get info for a specific sample
    print("Example 1: Get info for sample 0")
    info = mapper.get_sample_info(0)
    print(f"  Sample {info['sample_id']} is in cluster {info['cluster']}")
    print(f"  Atomic position in image: ({info['atomic_x']}, {info['atomic_y']})")
    print(f"  vPCF origin: ({info['vpcf_origin_x']}, {info['vpcf_origin_y']})")
    print()
    
    # Example 2: Get all samples in a cluster
    print("Example 2: Get all samples in cluster 0")
    cluster_0_df = mapper.get_cluster_samples(0)
    print(f"  Found {len(cluster_0_df)} samples in cluster 0")
    print(cluster_0_df.head())
    print()
    
    # Example 3: Get samples in a spatial region
    print("Example 3: Get samples with atomic positions in region [1600-1650, 1300-1400]")
    region_df = mapper.get_samples_in_region(1600, 1650, 1300, 1400)
    print(f"  Found {len(region_df)} samples in region")
    print(region_df.head())
    print()
    
    # Example 4: Export all cluster assignments with coordinates
    print("Example 4: Export all data to CSV")
    output_file = workspace_root / "results" / f"{mapper.method}_cluster_coordinates.csv"
    full_df = mapper.export_cluster_coordinates(str(output_file))
    print(f"  Exported {len(full_df)} samples")
    print()
    
    # Example 5: Get cluster statistics
    print("Example 5: Cluster spatial statistics")
    stats = mapper.get_cluster_statistics()
    for cluster_name, cluster_stats in stats.items():
        print(f"  {cluster_name}:")
        print(f"    Samples: {cluster_stats['n_samples']}")
        print(f"    X range: [{cluster_stats['atomic_x_min']:.0f}, {cluster_stats['atomic_x_max']:.0f}]")
        print(f"    Y range: [{cluster_stats['atomic_y_min']:.0f}, {cluster_stats['atomic_y_max']:.0f}]")
