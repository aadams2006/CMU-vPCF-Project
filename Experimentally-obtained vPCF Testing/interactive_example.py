"""
Quick interactive example showing how to use the vPCF cluster visual mapper.
Can be run as a script or adapted for use in Jupyter notebooks.

This example demonstrates the complete workflow:
1. Load the H5 file
2. Load inspection results
3. Explore clusters
4. Export for visual analysis
5. Compare models
"""

from pathlib import Path
from vpcf_cluster_visual_mapping import VPCFClusterMapper
import numpy as np


def print_section(title):
    """Print a nicely formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def interactive_demo():
    """Run an interactive demonstration of the mapper."""
    
    # =========================================================================
    # STEP 1: Initialize the mapper
    # =========================================================================
    print_section("STEP 1: Initialize Mapper")
    
    h5_file = Path("data/vPCF_test_2.h5")
    print(f"Loading H5 file: {h5_file}")
    
    if not h5_file.exists():
        print(f"ERROR: {h5_file} not found!")
        return
    
    mapper = VPCFClusterMapper(h5_file, verbose=False)
    print(f"Success! Loaded {mapper.n_samples} samples")
    print(f"Each vPCF image: {mapper.vpcf_shape}")
    
    # =========================================================================
    # STEP 2: Load inspection results (if available)
    # =========================================================================
    print_section("STEP 2: Load Inspection Results")
    
    dec_file = Path("results/dec/vpcf/DEC_detailed_cluster_report.csv")
    idec_file = Path("results/idec/vpcf/IDEC_detailed_cluster_report.csv")
    
    if dec_file.exists():
        print(f"Loading DEC results: {dec_file}")
        mapper.load_inspection_results(dec_file, source_name="dec")
        dec_stats = mapper.get_summary_statistics(source="dec")
        print(f"  DEC clusters: {dec_stats['n_clusters']}")
        print(f"  Samples: {dec_stats['total_samples']}")
        print(f"  Cluster sizes: {dec_stats['cluster_sizes']}")
    else:
        print(f"Note: DEC results not found at {dec_file}")
        print("      Run the training pipeline first to generate cluster results.")
    
    if idec_file.exists():
        print(f"\nLoading IDEC results: {idec_file}")
        mapper.load_inspection_results(idec_file, source_name="idec")
        idec_stats = mapper.get_summary_statistics(source="idec")
        print(f"  IDEC clusters: {idec_stats['n_clusters']}")
        print(f"  Samples: {idec_stats['total_samples']}")
        print(f"  Cluster sizes: {idec_stats['cluster_sizes']}")
    else:
        print(f"Note: IDEC results not found at {idec_file}")
    
    if "dec" not in mapper.inspection_results:
        print("\nNo inspection results loaded. Skipping remaining steps.")
        print("To proceed, you need to generate inspection results from your trained models.")
        return
    
    source = "dec"  # Use DEC results for examples
    
    # =========================================================================
    # STEP 3: Explore a specific sample
    # =========================================================================
    print_section("STEP 3: Explore a Specific Sample")
    
    sample_id = 0
    print(f"Retrieving data for sample {sample_id}...")
    
    vpcf_data = mapper.get_vpcf_by_sample_id(sample_id, return_metadata=True)
    
    print(f"vPCF data retrieved:")
    print(f"  vPCF image shape: {vpcf_data['vpcf_image'].shape}")
    print(f"  vPCF intensity - min: {vpcf_data['vpcf_image'].min():.4f}, max: {vpcf_data['vpcf_image'].max():.4f}, mean: {vpcf_data['vpcf_image'].mean():.4f}")
    
    if 'atomic_positions' in vpcf_data:
        atomic_pos = vpcf_data['atomic_positions']
        print(f"  Atomic position (scan coords): ({atomic_pos[0]:.2f}, {atomic_pos[1]:.2f})")
    
    if 'vpcf_origin' in vpcf_data:
        origin = vpcf_data['vpcf_origin']
        print(f"  vPCF origin: ({origin[0]:.2f}, {origin[1]:.2f})")
    
    # Get cluster assignment
    sample_cluster_data = mapper.inspection_results[source][
        mapper.inspection_results[source]['sample_idx'] == sample_id
    ]
    if len(sample_cluster_data) > 0:
        cluster_id = sample_cluster_data['predicted_cluster'].values[0]
        print(f"  Cluster assignment (DEC): {cluster_id}")
    
    # =========================================================================
    # STEP 4: Explore a cluster
    # =========================================================================
    print_section("STEP 4: Explore a Cluster")
    
    cluster_id = 0
    cluster_info = mapper.get_cluster_info(cluster_id, source=source)
    
    print(f"Cluster {cluster_id} information:")
    print(f"  Sample count: {cluster_info['sample_count']}")
    print(f"  Percentage of data: {cluster_info['percentage']:.2f}%")
    print(f"  Sample IDs (first 10): {cluster_info['sample_ids'][:10]}")
    
    # Get some statistics about the cluster
    print(f"\nCluster vPCF statistics (first 5 samples):")
    for idx, sid in enumerate(cluster_info['sample_ids'][:5]):
        vpcf = mapper.get_vpcf_by_sample_id(sid)['vpcf_image']
        print(f"  Sample {sid}: mean={vpcf.mean():.4f}, std={vpcf.std():.4f}, max={vpcf.max():.4f}")
    
    # =========================================================================
    # STEP 5: Create mapping report
    # =========================================================================
    print_section("STEP 5: Create Mapping Report")
    
    print("Creating mapping report with atomic position information...")
    report = mapper.create_cluster_mapping_report(
        source=source,
        output_dir="results/analysis"
    )
    
    print(f"Report created! Preview of first 5 rows:")
    print(report.head())
    
    # =========================================================================
    # STEP 6: Export cluster for visual analysis
    # =========================================================================
    print_section("STEP 6: Export Cluster for Visual Analysis")
    
    cluster_id = 0
    max_samples = 6
    output_dir = f"results/visual_analysis/cluster_{cluster_id}"
    
    print(f"Exporting cluster {cluster_id} to {output_dir}...")
    print(f"  Max samples: {max_samples}")
    print(f"  Save images: Yes (PNG montage)")
    print(f"  Save metadata: Yes (CSV)")
    
    cluster_export = mapper.export_cluster_for_visual_analysis(
        cluster_id=cluster_id,
        output_dir=output_dir,
        source=source,
        max_samples=max_samples,
        save_images=True,
        save_csv=True
    )
    
    print(f"\nExport summary:")
    print(f"  Cluster ID: {cluster_export['cluster_id']}")
    print(f"  Total samples exported: {len(cluster_export['samples'])}")
    print(f"  Output directory: {output_dir}")
    
    # =========================================================================
    # STEP 7: Model comparison (if IDEC available)
    # =========================================================================
    if "idec" in mapper.inspection_results:
        print_section("STEP 7: Compare DEC vs IDEC")
        
        from vpcf_visual_analysis_integration import compare_dec_vs_idec_clusters
        
        print("Comparing DEC and IDEC cluster assignments...")
        comparison = compare_dec_vs_idec_clusters(
            h5_filepath=h5_file,
            dec_results_path=dec_file,
            idec_results_path=idec_file,
            output_dir="results/comparison_example"
        )
        
        # Show samples with disagreement
        disagreements = comparison[~comparison['clusters_match']]
        print(f"\nDEC vs IDEC comparison:")
        print(f"  Total samples: {len(comparison)}")
        print(f"  Matching assignments: {(comparison['clusters_match']).sum()}")
        print(f"  Disagreements: {len(disagreements)}")
        
        if len(disagreements) > 0:
            print(f"\nFirst 5 disagreements:")
            print(disagreements.head())
    
    # =========================================================================
    # STEP 8: Summary
    # =========================================================================
    print_section("STEP 8: Summary")
    
    print("Workflow completed successfully!")
    print("\nWhat you can do next:")
    print("  1. Change 'cluster_id' to explore other clusters")
    print("  2. Increase 'max_samples' to see more vPCF patterns")
    print("  3. Export all clusters using create_cluster_visual_analysis()")
    print("  4. Visualize the generated montages and examine vPCF patterns")
    print("  5. Check the metadata CSV files for atomic positions")
    print("  6. Use vPCF patterns to interpret your DEC/IDEC clustering results")
    
    print("\nOutput files created:")
    print(f"  - {output_dir}/cluster_{cluster_id}_vpcf_montage_dec.png")
    print(f"  - {output_dir}/cluster_{cluster_id}_samples_metadata.csv")
    print(f"  - results/analysis/{source}_sample_to_cluster_mapping.csv")
    
    if "idec" in mapper.inspection_results:
        print(f"  - results/comparison_example/DEC_vs_IDEC_comparison.csv")
    
    print("\nFor more information, see VISUAL_MAPPING_GUIDE.md")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  vPCF CLUSTER VISUAL MAPPING - INTERACTIVE EXAMPLE")
    print("="*70)
    
    try:
        interactive_demo()
        print("\n" + "="*70)
        print("  Demo completed successfully!")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
