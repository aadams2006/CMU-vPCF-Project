"""
Example usage of the VPCFClusterMapper for visual analysis of cluster results.

This script demonstrates how to:
1. Load cluster inspection results from DEC/IDEC models
2. Map samples back to original vPCF images
3. Export clusters for visual analysis
4. Create comparison reports
"""

from pathlib import Path
from vpcf_cluster_visual_mapping import VPCFClusterMapper, create_cluster_visual_analysis
import os


def example_basic_usage():
    """Basic example: Load results and retrieve vPCF for a specific sample."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage - Retrieve vPCF by Sample ID")
    print("=" * 70)
    
    # Initialize mapper with H5 file
    h5_file = Path("data/Crop_HAADF_Projection_80pixels.h5")
    mapper = VPCFClusterMapper(h5_file, verbose=True)
    
    # Retrieve vPCF data and metadata for sample 0
    vpcf_data = mapper.get_vpcf_by_sample_id(
        sample_id=0,
        return_metadata=True
    )
    
    print(f"\nSample 0 vPCF shape: {vpcf_data['vpcf_image'].shape}")
    if 'atomic_positions' in vpcf_data:
        print(f"Atomic positions: {vpcf_data['atomic_positions']}")
    
    return mapper


def example_load_inspection_results(mapper):
    """Load DEC and IDEC inspection results."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Load Inspection Results")
    print("=" * 70)
    
    # Load DEC results (if available)
    dec_results_path = Path("results/dec/vpcf/DEC_detailed_cluster_report.csv")
    if dec_results_path.exists():
        dec_results = mapper.load_inspection_results(
            dec_results_path,
            source_name="dec"
        )
        print(f"\nDEC Results Summary:")
        print(mapper.get_summary_statistics(source="dec"))
    else:
        print(f"\nDEC results not found at {dec_results_path}")
        print("Note: You may need to run the training pipeline first to generate inspection results.")
    
    # Load IDEC results (if available)
    idec_results_path = Path("results/idec/vpcf/IDEC_detailed_cluster_report.csv")
    if idec_results_path.exists():
        idec_results = mapper.load_inspection_results(
            idec_results_path,
            source_name="idec"
        )
        print(f"\nIDEC Results Summary:")
        print(mapper.get_summary_statistics(source="idec"))
    else:
        print(f"\nIDEC results not found at {idec_results_path}")


def example_get_cluster_samples(mapper):
    """Get all samples in a specific cluster."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Get Samples in a Cluster")
    print("=" * 70)
    
    source = "dec" if "dec" in mapper.inspection_results else None
    if source is None:
        print("No inspection results loaded. Skipping this example.")
        return
    
    # Get info for cluster 0
    cluster_info = mapper.get_cluster_info(cluster_id=0, source=source)
    
    print(f"\nCluster 0 Information:")
    print(f"  Sample count: {cluster_info['sample_count']}")
    print(f"  Percentage of data: {cluster_info['percentage']:.2f}%")
    print(f"  First 10 sample IDs: {cluster_info['sample_ids'][:10]}")


def example_create_mapping_report(mapper):
    """Create a detailed mapping report."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Create Cluster-to-vPCF Mapping Report")
    print("=" * 70)
    
    source = "dec" if "dec" in mapper.inspection_results else None
    if source is None:
        print("No inspection results loaded. Skipping this example.")
        return
    
    # Create mapping report with atomic position info
    report = mapper.create_cluster_mapping_report(
        source=source,
        output_dir="results/analysis"
    )
    
    print(f"\nMapping Report created with {len(report)} rows")
    print("First few rows:")
    print(report.head(10))


def example_export_cluster_for_analysis(mapper):
    """Export a cluster for visual analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Export Cluster for Visual Analysis")
    print("=" * 70)
    
    source = "dec" if "dec" in mapper.inspection_results else None
    if source is None:
        print("No inspection results loaded. Skipping this example.")
        return
    
    # Export cluster 0 with limited samples
    cluster_info = mapper.export_cluster_for_visual_analysis(
        cluster_id=0,
        output_dir="results/visual_analysis/cluster_0",
        source=source,
        max_samples=6,
        save_images=True,
        save_csv=True
    )
    
    print(f"\nExported cluster information:")
    print(f"  Cluster ID: {cluster_info['cluster_id']}")
    print(f"  Total samples in cluster: {cluster_info['total_samples']}")
    print(f"  Exported: {len(cluster_info['samples'])} samples")


def example_map_to_stem(mapper):
    """Attempt to map vPCF back to STEM image."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Map vPCF to STEM (if DM4 available)")
    print("=" * 70)
    
    sample_id = 0
    dm3_file = Path("data/Crop_HAADF_Projection.dm4")
    
    mapping = mapper.map_to_stem_if_available(
        sample_id=sample_id,
        dm3_filepath=dm3_file if dm3_file.exists() else None
    )
    
    print(f"\nvPCF to STEM Mapping for Sample {sample_id}:")
    print(f"  vPCF shape: {mapping['vpcf_data'].shape}")
    print(f"  Mapping info: {mapping['mapping_info']}")


def example_create_all_cluster_analysis():
    """Create analysis for all clusters at once."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Batch Analysis - All Clusters")
    print("=" * 70)
    
    h5_file = Path("data/Crop_HAADF_Projection_80pixels.h5")
    
    # Check if DEC results exist
    dec_results_path = Path("results/dec/vpcf/DEC_detailed_cluster_report.csv")
    if dec_results_path.exists():
        print("Creating visual analysis for all DEC clusters...")
        create_cluster_visual_analysis(
            h5_filepath=h5_file,
            inspection_csv=dec_results_path,
            output_dir="results/dec_visual_analysis",
            source_name="dec",
            max_samples_per_cluster=5
        )
        print("✓ DEC analysis complete")
    else:
        print(f"DEC results not found at {dec_results_path}")
    
    # Check if IDEC results exist
    idec_results_path = Path("results/idec/vpcf/IDEC_detailed_cluster_report.csv")
    if idec_results_path.exists():
        print("\nCreating visual analysis for all IDEC clusters...")
        create_cluster_visual_analysis(
            h5_filepath=h5_file,
            inspection_csv=idec_results_path,
            output_dir="results/idec_visual_analysis",
            source_name="idec",
            max_samples_per_cluster=5
        )
        print("✓ IDEC analysis complete")
    else:
        print(f"IDEC results not found at {idec_results_path}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("vPCF CLUSTER VISUAL MAPPING - USAGE EXAMPLES")
    print("=" * 70)
    
    # Initialize mapper
    h5_file = Path("data/Crop_HAADF_Projection_80pixels.h5")
    if not h5_file.exists():
        print(f"\nError: H5 file not found at {h5_file}")
        print("Please ensure the Crop_HAADF_Projection_80pixels.h5 file is in the data/ directory.")
        return
    
    mapper = example_basic_usage()
    example_load_inspection_results(mapper)
    example_get_cluster_samples(mapper)
    example_create_mapping_report(mapper)
    example_export_cluster_for_analysis(mapper)
    example_map_to_stem(mapper)
    
    # Batch analysis (optional, can be slow for large datasets)
    print("\n" + "=" * 70)
    print("NOTE: To run batch analysis on all clusters, uncomment this line:")
    print("example_create_all_cluster_analysis()")
    print("=" * 70)


if __name__ == "__main__":
    main()
