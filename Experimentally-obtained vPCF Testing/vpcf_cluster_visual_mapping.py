"""
Utilities for mapping cluster results back to original vPCF images and STEM data.

This module provides functions to:
- Load cluster inspection results (sample_id to cluster mappings)
- Retrieve vPCF images from the H5 file by sample index
- Extract atomic positions (scan coordinates) from the H5 file
- Map vPCF back to STEM images if available
- Create visual analysis reports with sample images, vPCF, and cluster info

Usage:
    # Load inspection results and create mapping
    mapper = VPCFClusterMapper("path/to/vPCF_test_2.h5")
    
    # For DEC results
    dec_results = mapper.load_inspection_results(
        "results/dec/vpcf/DEC_detailed_cluster_report.csv"
    )
    
    # Retrieve vPCF and atomic info for specific sample
    vpcf_data = mapper.get_vpcf_by_sample_id(sample_id=0)
    
    # Get all samples in a specific cluster
    cluster_samples = mapper.get_cluster_samples(cluster_id=1, source="dec")
    
    # Export cluster data for visual analysis
    mapper.export_cluster_for_visual_analysis(
        cluster_id=1,
        output_dir="visual_analysis/cluster_1",
        source="dec"
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class VPCFClusterMapper:
    """
    Maps cluster assignments back to original vPCF images and STEM data.
    """
    
    def __init__(self, h5_filepath: Union[str, Path], verbose: bool = True):
        """
        Initialize the mapper with an H5 vPCF file.
        
        Parameters
        ----------
        h5_filepath : str or Path
            Path to the vPCF_test_2.h5 file containing raw vPCF data
        verbose : bool
            Whether to print debug information
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required. Install with: pip install h5py")
        
        self.h5_filepath = Path(h5_filepath)
        self.verbose = verbose
        
        if not self.h5_filepath.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_filepath}")
        
        # Load metadata about the H5 file
        self._load_h5_metadata()
        
        # Store loaded inspection results
        self.inspection_results: Dict[str, pd.DataFrame] = {}
        
        if self.verbose:
            print(f"Initialized VPCFClusterMapper with {self.n_samples} samples")
    
    def _load_h5_metadata(self) -> None:
        """Load basic metadata from the H5 file."""
        with h5py.File(self.h5_filepath, 'r') as f:
            if "experiments" in f:
                grp = f["experiments"]
                self.n_samples = grp["vpcf_images"].shape[0]
                self.vpcf_shape = grp["vpcf_images"].shape[1:]
                self.has_atomic_positions = "atomic_positions" in grp
                self.has_vpcf_origin = "vpcf_origin" in grp
            else:
                raise ValueError("H5 file does not have 'experiments' group")
    
    def load_inspection_results(
        self,
        csv_filepath: Union[str, Path],
        source_name: str = "model"
    ) -> pd.DataFrame:
        """
        Load inspection results from a detailed cluster report CSV.
        
        Parameters
        ----------
        csv_filepath : str or Path
            Path to the inspection results CSV (e.g., DEC_detailed_cluster_report.csv)
        source_name : str
            Name to store these results under (e.g., "dec", "idec")
            
        Returns
        -------
        pd.DataFrame
            The loaded inspection results
        """
        csv_filepath = Path(csv_filepath)
        if not csv_filepath.exists():
            raise FileNotFoundError(f"Inspection results file not found: {csv_filepath}")
        
        df = pd.read_csv(csv_filepath)
        self.inspection_results[source_name] = df
        
        if self.verbose:
            n_clusters = df['predicted_cluster'].nunique()
            print(f"Loaded inspection results '{source_name}': {len(df)} samples, {n_clusters} clusters")
        
        return df
    
    def get_vpcf_by_sample_id(
        self,
        sample_id: int,
        return_metadata: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve vPCF image and metadata for a specific sample.
        
        Parameters
        ----------
        sample_id : int
            Index of the sample in the H5 file (0-based)
        return_metadata : bool
            Whether to include atomic_positions and vpcf_origin
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'vpcf_image': The vPCF image (shape: vpcf_shape)
            - 'atomic_positions': Atomic coordinates (if return_metadata=True)
            - 'vpcf_origin': vPCF origin coordinates (if return_metadata=True)
        """
        if sample_id < 0 or sample_id >= self.n_samples:
            raise IndexError(f"Sample ID {sample_id} out of range [0, {self.n_samples})")
        
        data = {}
        
        with h5py.File(self.h5_filepath, 'r') as f:
            grp = f["experiments"]
            data['vpcf_image'] = grp["vpcf_images"][sample_id][:]
            
            if return_metadata:
                if self.has_atomic_positions:
                    data['atomic_positions'] = grp["atomic_positions"][sample_id][:]
                if self.has_vpcf_origin:
                    data['vpcf_origin'] = grp["vpcf_origin"][sample_id][:]
        
        return data
    
    def get_cluster_samples(
        self,
        cluster_id: int,
        source: str = "model"
    ) -> np.ndarray:
        """
        Get all sample IDs that belong to a specific cluster.
        
        Parameters
        ----------
        cluster_id : int
            The cluster ID to retrieve samples for
        source : str
            Which inspection results to use (must be loaded first)
            
        Returns
        -------
        np.ndarray
            Array of sample indices in the specified cluster
        """
        if source not in self.inspection_results:
            raise ValueError(f"No inspection results loaded for source '{source}'")
        
        df = self.inspection_results[source]
        mask = df['predicted_cluster'] == cluster_id
        sample_ids = df[mask]['sample_idx'].values if 'sample_idx' in df.columns else np.where(mask)[0]
        
        return sample_ids
    
    def get_cluster_info(
        self,
        cluster_id: int,
        source: str = "model"
    ) -> Dict[str, any]:
        """
        Get information about a cluster.
        
        Parameters
        ----------
        cluster_id : int
            The cluster ID
        source : str
            Which inspection results to use
            
        Returns
        -------
        dict
            Dictionary with cluster statistics
        """
        sample_ids = self.get_cluster_samples(cluster_id, source=source)
        
        return {
            'cluster_id': cluster_id,
            'sample_count': len(sample_ids),
            'percentage': (len(sample_ids) / self.n_samples) * 100,
            'sample_ids': sample_ids
        }
    
    def create_cluster_mapping_report(
        self,
        source: str = "model",
        output_dir: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Create a detailed report mapping sample IDs to clusters with vPCF metadata.
        
        Parameters
        ----------
        source : str
            Which inspection results to use
        output_dir : str or Path, optional
            Directory to save the report CSV
            
        Returns
        -------
        pd.DataFrame
            Report with sample_id, cluster_id, and atomic position info
        """
        if source not in self.inspection_results:
            raise ValueError(f"No inspection results loaded for source '{source}'")
        
        df = self.inspection_results[source].copy()
        
        # Ensure sample_idx column exists
        if 'sample_idx' not in df.columns:
            df['sample_idx'] = np.arange(len(df))
        
        # Extract atomic positions for additional context
        if self.has_atomic_positions:
            atomic_pos_list = []
            for sample_id in df['sample_idx']:
                try:
                    vpcf_data = self.get_vpcf_by_sample_id(sample_id, return_metadata=True)
                    if 'atomic_positions' in vpcf_data:
                        pos = vpcf_data['atomic_positions']
                        atomic_pos_list.append(f"({pos[0]:.2f}, {pos[1]:.2f})")
                    else:
                        atomic_pos_list.append("N/A")
                except Exception as e:
                    if self.verbose:
                        print(f"Error retrieving atomic positions for sample {sample_id}: {e}")
                    atomic_pos_list.append("Error")
            
            df['atomic_positions'] = atomic_pos_list
        
        # Rename cluster column for clarity
        if 'predicted_cluster' in df.columns:
            df = df.rename(columns={'predicted_cluster': 'cluster_id'})
        
        # Save if output directory specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{source}_sample_to_cluster_mapping.csv"
            df.to_csv(output_file, index=False)
            if self.verbose:
                print(f"Saved mapping report to: {output_file}")
        
        return df
    
    def export_cluster_for_visual_analysis(
        self,
        cluster_id: int,
        output_dir: Union[str, Path],
        source: str = "model",
        max_samples: Optional[int] = None,
        save_images: bool = True,
        save_csv: bool = True
    ) -> Dict[str, any]:
        """
        Export all samples from a cluster for visual analysis.
        
        Creates:
        - CSV with sample metadata
        - PNG montage of vPCF images (if matplotlib available)
        
        Parameters
        ----------
        cluster_id : int
            The cluster to export
        output_dir : str or Path
            Directory to save analysis files
        source : str
            Which inspection results to use
        max_samples : int, optional
            Maximum number of samples to export. If None, export all.
        save_images : bool
            Whether to save vPCF images as PNG
        save_csv : bool
            Whether to save CSV with metadata
            
        Returns
        -------
        dict
            Metadata about the exported cluster
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get cluster samples
        sample_ids = self.get_cluster_samples(cluster_id, source=source)
        
        if max_samples is not None:
            sample_ids = sample_ids[:max_samples]
        
        cluster_info = {
            'cluster_id': cluster_id,
            'source': source,
            'total_samples': len(sample_ids),
            'samples': []
        }
        
        # Export vPCF images
        if save_images and HAS_MATPLOTLIB:
            self._save_cluster_images(cluster_id, sample_ids, output_dir, source)
        
        # Create and save metadata CSV
        if save_csv:
            metadata_rows = []
            for idx, sample_id in enumerate(sample_ids):
                try:
                    vpcf_data = self.get_vpcf_by_sample_id(sample_id, return_metadata=True)
                    
                    row = {
                        'index_in_cluster': idx,
                        'sample_id': sample_id,
                        'cluster_id': cluster_id,
                        'vpcf_shape': str(vpcf_data['vpcf_image'].shape)
                    }
                    
                    if 'atomic_positions' in vpcf_data:
                        pos = vpcf_data['atomic_positions']
                        row['atomic_x'] = float(pos[0])
                        row['atomic_y'] = float(pos[1])
                    
                    if 'vpcf_origin' in vpcf_data:
                        origin = vpcf_data['vpcf_origin']
                        row['origin_x'] = float(origin[0])
                        row['origin_y'] = float(origin[1])
                    
                    metadata_rows.append(row)
                    cluster_info['samples'].append(row)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing sample {sample_id}: {e}")
            
            # Save CSV
            if metadata_rows:
                metadata_df = pd.DataFrame(metadata_rows)
                csv_path = output_dir / f"cluster_{cluster_id}_samples_metadata.csv"
                metadata_df.to_csv(csv_path, index=False)
                
                if self.verbose:
                    print(f"Saved cluster metadata to: {csv_path}")
        
        return cluster_info
    
    def _save_cluster_images(
        self,
        cluster_id: int,
        sample_ids: np.ndarray,
        output_dir: Path,
        source: str,
        samples_per_row: int = 4
    ) -> None:
        """
        Save vPCF images from a cluster as a montage.
        
        Parameters
        ----------
        cluster_id : int
            Cluster ID
        sample_ids : np.ndarray
            Array of sample indices in the cluster
        output_dir : Path
            Output directory
        source : str
            Name of the source model
        samples_per_row : int
            Number of samples per row in montage
        """
        n_samples = len(sample_ids)
        n_rows = (n_samples + samples_per_row - 1) // samples_per_row
        
        fig, axes = plt.subplots(
            n_rows, samples_per_row,
            figsize=(4 * samples_per_row, 4 * n_rows)
        )
        
        if n_rows == 1 and samples_per_row == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or samples_per_row == 1:
            axes = axes.reshape(n_rows, samples_per_row)
        
        for idx, sample_id in enumerate(sample_ids):
            row = idx // samples_per_row
            col = idx % samples_per_row
            ax = axes[row, col]
            
            try:
                vpcf_data = self.get_vpcf_by_sample_id(sample_id, return_metadata=False)
                image = vpcf_data['vpcf_image']
                
                ax.imshow(image, cmap='viridis')
                ax.set_title(f"Sample {sample_id}\nCluster {cluster_id}")
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\nSample {sample_id}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_samples, n_rows * samples_per_row):
            row = idx // samples_per_row
            col = idx % samples_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_file = output_dir / f"cluster_{cluster_id}_vpcf_montage_{source}.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Saved cluster image montage to: {output_file}")
    
    def map_to_stem_if_available(
        self,
        sample_id: int,
        dm3_filepath: Optional[Union[str, Path]] = None
    ) -> Dict[str, any]:
        """
        Attempt to map a vPCF sample back to STEM image data.
        
        This function uses atomic_positions as scan coordinates to locate
        the sample in the STEM image if a DM3 file is provided.
        
        Parameters
        ----------
        sample_id : int
            The vPCF sample ID
        dm3_filepath : str or Path, optional
            Path to the DM3 STEM file. If not provided, only returns
            vPCF data and coordinates.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'vpcf_data': The vPCF image
            - 'atomic_positions': Scan coordinates (x, y)
            - 'stem_data': STEM image data (if dm3_filepath provided and file exists)
            - 'stem_crop': Cropped region around the position (if available)
        """
        result = {'vpcf_data': {}, 'stem_data': None, 'mapping_info': {}}
        
        # Get vPCF and atomic position
        vpcf_data = self.get_vpcf_by_sample_id(sample_id, return_metadata=True)
        result['vpcf_data'] = vpcf_data['vpcf_image']
        
        if 'atomic_positions' in vpcf_data:
            atomic_pos = vpcf_data['atomic_positions']
            result['mapping_info']['atomic_positions'] = atomic_pos
            result['mapping_info']['atomic_positions_desc'] = (
                f"Scan position: ({atomic_pos[0]:.2f}, {atomic_pos[1]:.2f})"
            )
        
        if 'vpcf_origin' in vpcf_data:
            result['mapping_info']['vpcf_origin'] = vpcf_data['vpcf_origin']
        
        # Try to load STEM data if DM3 filepath provided
        if dm3_filepath is not None:
            dm3_path = Path(dm3_filepath)
            if dm3_path.exists():
                try:
                    # Try using hyperspy or ncempy
                    try:
                        import hyperspy.api as hs
                        signal = hs.load(str(dm3_path))
                        stem_image = signal.data
                        result['stem_data'] = stem_image
                        result['mapping_info']['stem_loaded'] = True
                    except ImportError:
                        try:
                            import ncempy.io as nio
                            dm3_data = nio.dm.dmReader(str(dm3_path))
                            stem_image = dm3_data['data']
                            result['stem_data'] = stem_image
                            result['mapping_info']['stem_loaded'] = True
                        except ImportError:
                            result['mapping_info']['stem_loaded'] = False
                            result['mapping_info']['note'] = (
                                "Neither hyperspy nor ncempy available. "
                                "Install with: pip install hyperspy or pip install ncempy"
                            )
                except Exception as e:
                    result['mapping_info']['stem_load_error'] = str(e)
            else:
                result['mapping_info']['note'] = f"DM3 file not found: {dm3_path}"
        
        return result
    
    def get_summary_statistics(
        self,
        source: str = "model"
    ) -> Dict[str, any]:
        """
        Get summary statistics about loaded inspection results.
        
        Parameters
        ----------
        source : str
            Which inspection results to summarize
            
        Returns
        -------
        dict
            Summary statistics
        """
        if source not in self.inspection_results:
            raise ValueError(f"No inspection results loaded for source '{source}'")
        
        df = self.inspection_results[source]
        cluster_col = 'predicted_cluster' if 'predicted_cluster' in df.columns else 'cluster_id'
        
        unique_clusters = df[cluster_col].unique()
        cluster_sizes = df[cluster_col].value_counts().to_dict()
        
        return {
            'source': source,
            'total_samples': len(df),
            'n_clusters': len(unique_clusters),
            'cluster_sizes': cluster_sizes,
            'h5_file': str(self.h5_filepath),
            'n_samples_in_h5': self.n_samples,
            'vpcf_shape': self.vpcf_shape
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_cluster_visual_analysis(
    h5_filepath: Union[str, Path],
    inspection_csv: Union[str, Path],
    output_dir: Union[str, Path],
    source_name: str = "model",
    max_samples_per_cluster: Optional[int] = 10
) -> None:
    """
    Create visual analysis reports for all clusters.
    
    This is a convenience wrapper that:
    1. Loads the H5 file
    2. Loads the inspection results
    3. Exports each cluster for visual analysis
    
    Parameters
    ----------
    h5_filepath : str or Path
        Path to vPCF_test_2.h5
    inspection_csv : str or Path
        Path to inspection results CSV
    output_dir : str or Path
        Output directory for analysis results
    source_name : str
        Name of the model (dec, idec, etc)
    max_samples_per_cluster : int, optional
        Maximum samples to export per cluster
    """
    # Initialize mapper
    mapper = VPCFClusterMapper(h5_filepath, verbose=True)
    
    # Load results
    mapper.load_inspection_results(inspection_csv, source_name=source_name)
    
    # Get cluster info
    stats = mapper.get_summary_statistics(source=source_name)
    print(f"\nProcessing {stats['n_clusters']} clusters with {stats['total_samples']} samples")
    
    # Export each cluster
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for cluster_id in sorted(stats['cluster_sizes'].keys()):
        cluster_dir = output_dir / f"cluster_{cluster_id}"
        mapper.export_cluster_for_visual_analysis(
            cluster_id=cluster_id,
            output_dir=cluster_dir,
            source=source_name,
            max_samples=max_samples_per_cluster,
            save_images=True,
            save_csv=True
        )


if __name__ == "__main__":
    # Example usage
    print("vPCF Cluster Visual Mapping module loaded successfully")
    print("Use VPCFClusterMapper class for detailed analysis")
