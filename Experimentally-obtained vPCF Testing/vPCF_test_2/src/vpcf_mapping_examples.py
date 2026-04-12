"""
Practical examples and usage guides for vPCF to Image mapping.

This module demonstrates various ways to use the VPCFToImageMapper to:
1. Query specific vPCF samples
2. Analyze cluster distributions
3. Create spatial visualizations
4. Export data for downstream analysis
"""

from pathlib import Path
import pandas as pd
import numpy as np
from vpcf_to_image_mapper import VPCFToImageMapper


def example_1_query_single_vpcf():
    """
    Example 1: Query a specific vPCF sample to get its cluster and image coordinates.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Query Single vPCF Sample")
    print("="*70)
    
    workspace_root = Path(__file__).parent.parent
    mapper = VPCFToImageMapper(
        metadata_file=str(workspace_root / "results" / "all_samples_metadata.json"),
        clustering_results_dir=str(workspace_root / "results" / "h5_only" / "dec"),
        method='dec'
    )
    
    # Get info for vPCF sample 100
    sample_id = 100
    info = mapper.get_sample_info(sample_id)
    
    print(f"\nSample {sample_id} Information:")
    print(f"  - Cluster: {info['cluster']}")
    print(f"  - Atomic position in original image: ({info['atomic_x']:.1f}, {info['atomic_y']:.1f})")
    print(f"  - vPCF window origin: ({info['vpcf_origin_x']:.1f}, {info['vpcf_origin_y']:.1f})")
    print(f"  - vPCF window size: {info['vpcf_shape']}")
    print(f"  - Crystal label: {info['crystal_label']}")
    
    print("\n⚠️  Usage: Use these coordinates to locate the vPCF in the original image!")


def example_2_find_all_vpcfs_in_cluster():
    """
    Example 2: Find all vPCF samples belonging to a specific cluster.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Find All vPCFs in a Cluster")
    print("="*70)
    
    workspace_root = Path(__file__).parent.parent
    mapper = VPCFToImageMapper(
        metadata_file=str(workspace_root / "results" / "all_samples_metadata.json"),
        clustering_results_dir=str(workspace_root / "results" / "h5_only" / "dec"),
        method='dec'
    )
    
    # Get all samples in cluster 1
    cluster_id = 1
    cluster_df = mapper.get_cluster_samples(cluster_id)
    
    print(f"\nCluster {cluster_id} contains {len(cluster_df)} samples")
    print("\nFirst 10 samples:")
    print(cluster_df[['sample_id', 'atomic_x', 'atomic_y']].head(10))
    
    print(f"\nSpatial extent of cluster {cluster_id}:")
    print(f"  X range: [{cluster_df['atomic_x'].min():.0f}, {cluster_df['atomic_x'].max():.0f}]")
    print(f"  Y range: [{cluster_df['atomic_y'].min():.0f}, {cluster_df['atomic_y'].max():.0f}]")
    print(f"  Center: ({cluster_df['atomic_x'].mean():.0f}, {cluster_df['atomic_y'].mean():.0f})")


def example_3_spatial_query():
    """
    Example 3: Find vPCF samples in a specific region of the original image.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Spatial Query - Find vPCFs in Image Region")
    print("="*70)
    
    workspace_root = Path(__file__).parent.parent
    mapper = VPCFToImageMapper(
        metadata_file=str(workspace_root / "results" / "all_samples_metadata.json"),
        clustering_results_dir=str(workspace_root / "results" / "h5_only" / "dec"),
        method='dec'
    )
    
    # Find all vPCFs with atomic positions in region [500-800, 600-900]
    x_min, x_max = 500, 800
    y_min, y_max = 600, 900
    
    region_df = mapper.get_samples_in_region(x_min, x_max, y_min, y_max)
    
    print(f"\nSearching region: X[{x_min}, {x_max}], Y[{y_min}, {y_max}]")
    print(f"Found {len(region_df)} vPCF samples in this region\n")
    
    # Show cluster distribution in this region
    cluster_counts = region_df['cluster'].value_counts().sort_index()
    print("Cluster distribution in region:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} samples")


def example_4_export_for_visualization():
    """
    Example 4: Export cluster assignments with coordinates for visualization.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Export Data for Visualization")
    print("="*70)
    
    workspace_root = Path(__file__).parent.parent
    mapper = VPCFToImageMapper(
        metadata_file=str(workspace_root / "results" / "all_samples_metadata.json"),
        clustering_results_dir=str(workspace_root / "results" / "h5_only" / "dec"),
        method='dec'
    )
    
    # Export full data
    output_file = workspace_root / "results" / "dec_cluster_coordinates_export.csv"
    full_df = mapper.export_cluster_coordinates(str(output_file))
    
    print(f"\nExported {len(full_df)} samples to: {output_file}")
    print("\nFirst 5 rows of exported data:")
    print(full_df.head())


def example_5_integration_with_vpcf_data():
    """
    Example 5: Integrate cluster info with vPCF feature data.
    
    This shows how to combine the cluster assignments with the actual vPCF
    feature data (e.g., from all_samples_vpcf_data.csv).
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Integrate with vPCF Feature Data")
    print("="*70)
    
    workspace_root = Path(__file__).parent.parent
    mapper = VPCFToImageMapper(
        metadata_file=str(workspace_root / "results" / "all_samples_metadata.json"),
        clustering_results_dir=str(workspace_root / "results" / "h5_only" / "dec"),
        method='dec'
    )
    
    # Load vPCF feature data
    vpcf_data_file = workspace_root / "results" / "all_samples_vpcf_data.csv"
    vpcf_df = pd.read_csv(vpcf_data_file)
    
    # Load cluster coordinates
    cluster_coords_df = mapper.export_cluster_coordinates()
    
    # Merge just the cluster info we need
    cluster_info = cluster_coords_df[['sample_id', 'cluster']]
    merged_df = vpcf_df.merge(cluster_info, on='sample_id', how='left')
    
    print(f"\nMerged vPCF features with cluster assignments")
    print(f"Total rows: {len(merged_df)}")
    print(f"\nColumns available (first 5):")
    for col in merged_df.columns[:5]:
        print(f"  - {col}")
    print(f"  ...")
    
    print(f"\nExample row:")
    print(merged_df.iloc[0][['sample_id', 'cluster', 'atomic_x', 'atomic_y']])


def example_6_identify_cluster_boundaries():
    """
    Example 6: Identify boundaries and relationships between clusters.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Cluster Spatial Analysis")
    print("="*70)
    
    workspace_root = Path(__file__).parent.parent
    mapper = VPCFToImageMapper(
        metadata_file=str(workspace_root / "results" / "all_samples_metadata.json"),
        clustering_results_dir=str(workspace_root / "results" / "h5_only" / "dec"),
        method='dec'
    )
    
    stats = mapper.get_cluster_statistics()
    
    print("\nCluster Spatial Statistics:")
    print("-" * 70)
    
    for cluster_name, cluster_stats in sorted(stats.items()):
        print(f"\n{cluster_name}:")
        print(f"  Samples: {cluster_stats['n_samples']}")
        print(f"  Center: ({cluster_stats['atomic_x_mean']:.0f}, {cluster_stats['atomic_y_mean']:.0f})")
        print(f"  Std Dev: (±{cluster_stats['atomic_x_std']:.0f}, ±{cluster_stats['atomic_y_std']:.0f})")
        print(f"  X range: [{cluster_stats['atomic_x_min']:.0f}, {cluster_stats['atomic_x_max']:.0f}] " +
              f"(span: {cluster_stats['atomic_x_max'] - cluster_stats['atomic_x_min']:.0f})")
        print(f"  Y range: [{cluster_stats['atomic_y_min']:.0f}, {cluster_stats['atomic_y_max']:.0f}] " +
              f"(span: {cluster_stats['atomic_y_max'] - cluster_stats['atomic_y_min']:.0f})")


def create_batch_lookup_script():
    """
    Example: Create a script that does batch lookups for many vPCF samples.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Batch Lookup Template")
    print("="*70)
    
    example_code = '''
# Batch lookup template: Given a list of vPCF sample IDs, get their info
from pathlib import Path
from vpcf_to_image_mapper import VPCFToImageMapper

workspace_root = Path("path/to/workspace")
mapper = VPCFToImageMapper(
    metadata_file=str(workspace_root / "results" / "all_samples_metadata.json"),
    clustering_results_dir=str(workspace_root / "results" / "h5_only" / "dec"),
    method='dec'
)

# List of vPCF samples you're interested in
sample_ids = [50, 100, 200, 500, 1000]

results = []
for sample_id in sample_ids:
    info = mapper.get_sample_info(sample_id)
    results.append({
        'sample_id': sample_id,
        'cluster': info['cluster'],
        'atomic_x': info['atomic_x'],
        'atomic_y': info['atomic_y'],
    })

# Convert to DataFrame for easy analysis
import pandas as pd
batch_df = pd.DataFrame(results)
print(batch_df)
'''
    
    print("\nTemplate for batch lookups:")
    print(example_code)


if __name__ == "__main__":
    example_1_query_single_vpcf()
    example_2_find_all_vpcfs_in_cluster()
    example_3_spatial_query()
    example_4_export_for_visualization()
    example_5_integration_with_vpcf_data()
    example_6_identify_cluster_boundaries()
    create_batch_lookup_script()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
