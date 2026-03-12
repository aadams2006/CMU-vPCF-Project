"""
Quick usage examples for the sample metadata lookup system.

Run this file: python sample_usage_examples.py
"""

from pathlib import Path
from sample_metadata_lookup import SampleMetadataManager
from vpcf_cluster_visual_mapping import VPCFClusterMapper


def example_1_basic_lookup():
    """Example 1: Look up a single sample by index."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Look up Sample 123")
    print("=" * 70)
    
    # Create manager
    manager = SampleMetadataManager("data/vPCF_test_2.h5", verbose=False)
    
    # Get all metadata for sample 123
    info = manager.get_sample_info(123)
    
    print(f"\nSample 123:")
    print(f"  Atomic position (scan coords): {info['atomic_positions']}")
    print(f"  vPCF origin: {info['vpcf_origin']}")
    print(f"  vPCF shape: {info['vpcf_shape']}")
    print(f"  Peak shape: {info['peaks_shape']}")


def example_2_using_mapper_directly():
    """Example 2: Use VPCFClusterMapper extended methods."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Using VPCFClusterMapper Extended Methods")
    print("=" * 70)
    
    mapper = VPCFClusterMapper("data/vPCF_test_2.h5", verbose=False)
    
    # Method 1: Get metadata without image
    metadata = mapper.get_sample_metadata(123, include_vpcf_image=False)
    print(f"\nUsing mapper.get_sample_metadata(123):")
    for key, val in metadata.items():
        print(f"  {key}: {val}")
    
    # Method 2: Get metadata WITH vPCF image
    metadata_with_image = mapper.get_sample_metadata(123, include_vpcf_image=True)
    print(f"\nWith image: vpcf_image shape = {metadata_with_image['vpcf_image'].shape}")
    
    # Method 3: Inspect H5 structure
    print(f"\nAvailable datasets in H5:")
    structure = mapper.inspect_h5_full_structure()
    for group_name, group_info in structure['groups'].items():
        print(f"  /{group_name}/")
        for ds_name in group_info['datasets'].keys():
            print(f"    - {ds_name}")


def example_3_batch_export():
    """Example 3: Export metadata for multiple samples to CSV."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Export Metadata for Cluster Samples")
    print("=" * 70)
    
    manager = SampleMetadataManager("data/vPCF_test_2.h5", verbose=False)
    
    # Export metadata for samples in a hypothetical cluster
    cluster_samples = [0, 5, 10, 15, 20, 42, 100, 123]
    
    result = manager.export_sample_metadata(
        sample_ids=cluster_samples,
        output_file="cluster_metadata_example.csv",
        include_images=False
    )
    
    print(f"\nExported {result['samples_exported']} samples to: {result['output_file']}")
    print(f"Columns: {result['columns']}")
    
    # Read and display the CSV
    import pandas as pd
    df = pd.read_csv("cluster_metadata_example.csv")
    print(f"\nFirst 3 rows of exported data:")
    print(df.head(3).to_string())
    
    # Clean up
    Path("cluster_metadata_example.csv").unlink()
    print("\n(Temporary file deleted)")


def example_4_filter_by_position():
    """Example 4: Find samples by atomic position criteria."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Filter Samples by Atomic Position")
    print("=" * 70)
    
    mapper = VPCFClusterMapper("data/vPCF_test_2.h5", verbose=False)
    
    # Find first few samples with atomic_x > 1600
    print("\nFinding samples where atomic_x > 1600...")
    samples = mapper.get_samples_by_metadata_filter(
        lambda sid: mapper.get_sample_metadata(sid)['atomic_positions'][0] > 1600
    )
    
    print(f"Found {len(samples)} samples")
    print(f"First 10: {samples[:10]}")


def example_5_summary_report():
    """Example 5: Print metadata availability summary."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Metadata Availability Summary")
    print("=" * 70)
    
    manager = SampleMetadataManager("data/vPCF_test_2.h5", verbose=False)
    
    # Print comprehensive summary
    manager.print_summary()
    
    # Access summary programmatically
    report = manager.create_summary_report()
    print(f"\nAccessing report programmatically:")
    for key, value in report.items():
        if key != 'crystal_types':
            print(f"  {key}: {value}")


def example_6_integration_with_clustering():
    """Example 6: Integrate with clustering results."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Integration with Clustering Results")
    print("=" * 70)
    
    mapper = VPCFClusterMapper("data/vPCF_test_2.h5", verbose=False)
    manager = SampleMetadataManager("data/vPCF_test_2.h5", verbose=False)
    
    # Simulated cluster: samples [0, 15, 42, 100]
    cluster_id = 0
    cluster_samples = [0, 15, 42, 100]
    
    print(f"\nAnalyzing Cluster {cluster_id}:")
    print(f"  Samples: {cluster_samples}")
    
    # Create metadata for cluster
    rows = []
    for sample_id in cluster_samples:
        info = manager.get_sample_info(sample_id)
        rows.append({
            'sample_id': sample_id,
            'atomic_x': info['atomic_positions'][0],
            'atomic_y': info['atomic_positions'][1],
            'vpcf_shape': info['vpcf_shape']
        })
    
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"\nCluster metadata:")
    print(df.to_string(index=False))
    
    # Calculate cluster statistics
    print(f"\nCluster statistics:")
    print(f"  Mean atomic position: ({df['atomic_x'].mean():.2f}, {df['atomic_y'].mean():.2f})")
    print(f"  Std atomic_x: {df['atomic_x'].std():.2f}")
    print(f"  Std atomic_y: {df['atomic_y'].std():.2f}")


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("Sample Metadata Lookup - Usage Examples")
    print("█" * 70)
    
    example_1_basic_lookup()
    example_2_using_mapper_directly()
    example_3_batch_export()
    example_4_filter_by_position()
    example_5_summary_report()
    example_6_integration_with_clustering()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
