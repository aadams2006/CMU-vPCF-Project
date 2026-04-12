"""
Visualization of vPCF clusters color-coded on the original image space.

This module creates visual representations of cluster distributions by:
- Creating a spatial heatmap of cluster assignments
- Generating scatter plots showing cluster locations
- Comparing DEC vs IDEC clustering results
- Creating side-by-side visualizations for comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from vpcf_to_image_mapper import VPCFToImageMapper


class ClusterVisualizer:
    """Visualize vPCF cluster assignments on original image coordinates."""
    
    def __init__(self, 
                 h5_file: str,
                 clustering_results_dir: str,
                 method: str = 'dec'):
        """
        Initialize visualizer with data.
        
        Parameters
        ----------
        h5_file : str
            Path to the H5 data file (e.g., Crop_HAADF_Projection_80pixels.h5)
        clustering_results_dir : str
            Path to clustering results directory
        method : str
            'dec' or 'idec'
        """
        self.mapper = VPCFToImageMapper(
            h5_file=h5_file,
            clustering_results_dir=clustering_results_dir,
            method=method
        )
        self.method = method
        
        # Generate color map for clusters
        n_clusters = self.mapper.n_clusters
        self.colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 3)))
    
    def plot_cluster_scatter(self, 
                             figsize: tuple = (12, 10),
                             point_size: int = 20,
                             alpha: float = 0.6,
                             title: str = None) -> plt.Figure:
        """
        Create scatter plot of clusters in image coordinate space.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        point_size : int
            Size of each point
        alpha : float
            Transparency (0-1)
        title : str
            Custom title
        
        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each cluster
        for cluster_id in range(self.mapper.n_clusters):
            cluster_df = self.mapper.get_cluster_samples(cluster_id)
            
            ax.scatter(
                cluster_df['atomic_x'],
                cluster_df['atomic_y'],
                c=[self.colors[cluster_id]],
                label=f'Cluster {cluster_id} (n={len(cluster_df)})',
                s=point_size,
                alpha=alpha,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel('X Position in Original Image', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position in Original Image', fontsize=12, fontweight='bold')
        
        if title is None:
            title = f'vPCF {self.method.upper()} Cluster Assignments'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_heatmap(self,
                            figsize: tuple = (12, 10),
                            resolution: tuple = (200, 200),
                            title: str = None) -> plt.Figure:
        """
        Create heatmap visualization of cluster distribution.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        resolution : tuple
            Grid resolution for heatmap
        title : str
            Custom title
        
        Returns
        -------
        plt.Figure
            Figure object
        """
        # Get all sample data
        full_df = self.mapper.export_cluster_coordinates()
        
        # Create a grid representing the image space
        x_min, x_max = full_df['atomic_x'].min(), full_df['atomic_x'].max()
        y_min, y_max = full_df['atomic_y'].min(), full_df['atomic_y'].max()
        
        # Create grid bins
        x_bins = np.linspace(x_min, x_max, resolution[0])
        y_bins = np.linspace(y_min, y_max, resolution[1])
        
        # Create figure with subplots for each cluster
        n_clusters = self.mapper.n_clusters
        fig, axes = plt.subplots(
            1, n_clusters,
            figsize=(figsize[0], figsize[1] // 2),
            squeeze=False
        )
        
        for cluster_id in range(n_clusters):
            ax = axes[0, cluster_id]
            cluster_df = self.mapper.get_cluster_samples(cluster_id)
            
            # Create heatmap for this cluster
            heatmap, xedges, yedges = np.histogram2d(
                cluster_df['atomic_x'],
                cluster_df['atomic_y'],
                bins=[x_bins, y_bins]
            )
            
            im = ax.imshow(
                heatmap.T,
                extent=[x_min, x_max, y_max, y_min],
                cmap='hot',
                aspect='auto',
                origin='upper'
            )
            
            ax.set_xlabel('X Position', fontsize=10)
            ax.set_ylabel('Y Position', fontsize=10)
            ax.set_title(f'Cluster {cluster_id}\n({len(cluster_df)} samples)',
                        fontsize=11, fontweight='bold')
            
            plt.colorbar(im, ax=ax, label='Sample Count')
        
        if title is None:
            title = f'vPCF {self.method.upper()} Cluster Density Distribution'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return fig
    
    def plot_combined_heatmap(self,
                             figsize: tuple = (14, 6),
                             resolution: tuple = (300, 300),
                             title: str = None) -> plt.Figure:
        """
        Create single combined heatmap showing all clusters with color coding.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        resolution : tuple
            Grid resolution
        title : str
            Custom title
        
        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get all data
        full_df = self.mapper.export_cluster_coordinates()
        
        x_min, x_max = full_df['atomic_x'].min(), full_df['atomic_x'].max()
        y_min, y_max = full_df['atomic_y'].min(), full_df['atomic_y'].max()
        
        # Create RGB image where each pixel color represents dominant cluster
        x_bins = np.linspace(x_min, x_max, resolution[0] + 1)
        y_bins = np.linspace(y_min, y_max, resolution[1] + 1)
        
        # Initialize image and count arrays
        image = np.zeros((resolution[1], resolution[0], 3), dtype=np.float32)
        counts = np.zeros((resolution[1], resolution[0]), dtype=np.float32)
        
        # For each cluster, add its color contribution
        for cluster_id in range(self.mapper.n_clusters):
            cluster_df = self.mapper.get_cluster_samples(cluster_id)
            
            # Create heatmap for this cluster
            heatmap, _, _ = np.histogram2d(
                cluster_df['atomic_x'],
                cluster_df['atomic_y'],
                bins=[x_bins, y_bins]
            )
            
            # Ensure correct shape
            heatmap = heatmap[:resolution[0], :resolution[1]]
            heatmap_t = heatmap.T
            
            # Update counts to know which cluster is dominant
            counts += heatmap_t
            
            # Add this cluster's color contribution
            for i in range(3):
                image[:, :, i] += heatmap_t * self.colors[cluster_id, i]
        
        # Normalize so colors don't blow out
        counts[counts == 0] = 1  # Avoid division by zero
        for i in range(3):
            image[:, :, i] = image[:, :, i] / counts
        
        # Display
        im = ax.imshow(
            image,
            extent=[x_min, x_max, y_max, y_min],
            aspect='auto',
            origin='upper'
        )
        
        ax.set_xlabel('X Position in Original Image', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position in Original Image', fontsize=12, fontweight='bold')
        
        if title is None:
            title = f'vPCF {self.method.upper()} Cluster Color-Coded Heatmap'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Create legend
        legend_elements = [
            mpatches.Patch(facecolor=self.colors[i], edgecolor='black',
                          label=f'Cluster {i}')
            for i in range(self.mapper.n_clusters)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def compare_dec_vs_idec(self,
                           h5_file: str,
                           figsize: tuple = (20, 8),
                           title: str = None) -> plt.Figure:
        """
        Side-by-side comparison of DEC vs IDEC clustering on same space.
        
        Parameters
        ----------
        h5_file : str
            Path to H5 data file
        figsize : tuple
            Figure size
        title : str
            Custom title
        
        Returns
        -------
        plt.Figure
            Figure object
        """
        # Create visualizers for both methods
        dec_vis = ClusterVisualizer(h5_file, 
                                   str(Path(self.mapper.clustering_results_dir).parent),
                                   method='dec')
        idec_vis = ClusterVisualizer(h5_file,
                                    str(Path(self.mapper.clustering_results_dir).parent),
                                    method='idec')
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # DEC plot
        ax_dec = axes[0]
        for cluster_id in range(dec_vis.mapper.n_clusters):
            cluster_df = dec_vis.mapper.get_cluster_samples(cluster_id)
            ax_dec.scatter(
                cluster_df['atomic_x'],
                cluster_df['atomic_y'],
                c=[dec_vis.colors[cluster_id]],
                label=f'Cluster {cluster_id} (n={len(cluster_df)})',
                s=20,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax_dec.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax_dec.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        ax_dec.set_title('DEC Clustering Results', fontsize=13, fontweight='bold')
        ax_dec.legend(loc='best', fontsize=9, framealpha=0.9)
        ax_dec.grid(True, alpha=0.3)
        ax_dec.invert_yaxis()
        
        # IDEC plot
        ax_idec = axes[1]
        for cluster_id in range(idec_vis.mapper.n_clusters):
            cluster_df = idec_vis.mapper.get_cluster_samples(cluster_id)
            ax_idec.scatter(
                cluster_df['atomic_x'],
                cluster_df['atomic_y'],
                c=[idec_vis.colors[cluster_id]],
                label=f'Cluster {cluster_id} (n={len(cluster_df)})',
                s=20,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax_idec.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax_idec.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        ax_idec.set_title('IDEC Clustering Results', fontsize=13, fontweight='bold')
        ax_idec.legend(loc='best', fontsize=9, framealpha=0.9)
        ax_idec.grid(True, alpha=0.3)
        ax_idec.invert_yaxis()
        
        if title is None:
            title = 'Comparison: DEC vs IDEC Clustering'
        fig.suptitle(title, fontsize=15, fontweight='bold')
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    workspace_root = Path(__file__).parent.parent
    h5_file = str(workspace_root / "data" / "Crop_HAADF_Projection_80pixels.h5")
    clustering_dec_dir = str(workspace_root / "results" / "h5_only" / "dec")
    clustering_idec_dir = str(workspace_root / "results" / "h5_only" / "idec")
    
    print("Creating cluster visualizations...\n")
    
    # =========================================================================
    # DEC Clustering Visualizations
    # =========================================================================
    print("DEC Visualizations:")
    print("-" * 70)
    
    dec_vis = ClusterVisualizer(h5_file, clustering_dec_dir, method='dec')
    
    # 1. Scatter plot
    print("1. Creating scatter plot...")
    fig_scatter = dec_vis.plot_cluster_scatter(
        title='DEC: vPCF Cluster Assignments in Original Image'
    )
    fig_scatter.savefig(
        workspace_root / "results" / "DEC_cluster_scatter.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("   Saved: DEC_cluster_scatter.png")
    
    # 2. Density heatmaps
    print("2. Creating density heatmaps...")
    fig_heatmap = dec_vis.plot_cluster_heatmap(
        title='DEC: Cluster Density Distribution'
    )
    fig_heatmap.savefig(
        workspace_root / "results" / "DEC_cluster_heatmap.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("   Saved: DEC_cluster_heatmap.png")
    
    # 3. Combined heatmap
    print("3. Creating combined color-coded heatmap...")
    fig_combined = dec_vis.plot_combined_heatmap(
        title='DEC: Color-Coded Cluster Heatmap'
    )
    fig_combined.savefig(
        workspace_root / "results" / "DEC_cluster_combined_heatmap.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("   Saved: DEC_cluster_combined_heatmap.png\n")
    
    # =========================================================================
    # IDEC Clustering Visualizations
    # =========================================================================
    print("IDEC Visualizations:")
    print("-" * 70)
    
    idec_vis = ClusterVisualizer(h5_file, clustering_idec_dir, method='idec')
    
    # 1. Scatter plot
    print("1. Creating scatter plot...")
    fig_scatter = idec_vis.plot_cluster_scatter(
        title='IDEC: vPCF Cluster Assignments in Original Image'
    )
    fig_scatter.savefig(
        workspace_root / "results" / "IDEC_cluster_scatter.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("   Saved: IDEC_cluster_scatter.png")
    
    # 2. Density heatmaps
    print("2. Creating density heatmaps...")
    fig_heatmap = idec_vis.plot_cluster_heatmap(
        title='IDEC: Cluster Density Distribution'
    )
    fig_heatmap.savefig(
        workspace_root / "results" / "IDEC_cluster_heatmap.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("   Saved: IDEC_cluster_heatmap.png")
    
    # 3. Combined heatmap
    print("3. Creating combined color-coded heatmap...")
    fig_combined = idec_vis.plot_combined_heatmap(
        title='IDEC: Color-Coded Cluster Heatmap'
    )
    fig_combined.savefig(
        workspace_root / "results" / "IDEC_cluster_combined_heatmap.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("   Saved: IDEC_cluster_combined_heatmap.png\n")
    
    # =========================================================================
    # Comparison
    # =========================================================================
    print("Comparison Visualization:")
    print("-" * 70)
    print("1. Creating DEC vs IDEC comparison...")
    
    # We'll manually create this comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # DEC
    ax = axes[0]
    for cluster_id in range(dec_vis.mapper.n_clusters):
        cluster_df = dec_vis.mapper.get_cluster_samples(cluster_id)
        ax.scatter(
            cluster_df['atomic_x'],
            cluster_df['atomic_y'],
            c=[dec_vis.colors[cluster_id]],
            label=f'Cluster {cluster_id} (n={len(cluster_df)})',
            s=30,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
    ax.set_xlabel('X Position in Original Image', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position in Original Image', fontsize=12, fontweight='bold')
    ax.set_title('DEC Clustering', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # IDEC
    ax = axes[1]
    for cluster_id in range(idec_vis.mapper.n_clusters):
        cluster_df = idec_vis.mapper.get_cluster_samples(cluster_id)
        ax.scatter(
            cluster_df['atomic_x'],
            cluster_df['atomic_y'],
            c=[idec_vis.colors[cluster_id]],
            label=f'Cluster {cluster_id} (n={len(cluster_df)})',
            s=30,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
    ax.set_xlabel('X Position in Original Image', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position in Original Image', fontsize=12, fontweight='bold')
    ax.set_title('IDEC Clustering', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    fig.suptitle('Comparison: DEC vs IDEC Clustering Results', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(
        workspace_root / "results" / "DEC_vs_IDEC_comparison.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("   Saved: DEC_vs_IDEC_comparison.png\n")
    
    print("="*70)
    print("All visualizations completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  DEC visualizations:")
    print("    - DEC_cluster_scatter.png (Scatter plot)")
    print("    - DEC_cluster_heatmap.png (Density heatmaps)")
    print("    - DEC_cluster_combined_heatmap.png (Color-coded heatmap)")
    print("  IDEC visualizations:")
    print("    - IDEC_cluster_scatter.png (Scatter plot)")
    print("    - IDEC_cluster_heatmap.png (Density heatmaps)")
    print("    - IDEC_cluster_combined_heatmap.png (Color-coded heatmap)")
    print("  Comparison:")
    print("    - DEC_vs_IDEC_comparison.png (Side-by-side)")
