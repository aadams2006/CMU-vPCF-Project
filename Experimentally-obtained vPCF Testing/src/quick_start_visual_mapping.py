#!/usr/bin/env python3
"""
Quick-start script for vPCF cluster visual mapping.

This script provides the fastest way to:
1. Map cluster assignments to vPCF images
2. Export clusters for visual analysis
3. Compare DEC and IDEC results

Usage:
    python quick_start_visual_mapping.py --help
    python quick_start_visual_mapping.py --export-clusters dec --max-samples 5
    python quick_start_visual_mapping.py --compare
"""

import argparse
from pathlib import Path
from typing import Optional


def export_clusters_command(
    model: str = "dec",
    max_samples: int = 5,
    output_dir: Optional[str] = None
):
    """Export all clusters for a model."""
    from vpcf_cluster_visual_mapping import VPCFClusterMapper, create_cluster_visual_analysis
    
    h5_file = Path("data/vPCF_test_2.h5")
    
    if model.lower() == "dec":
        results_file = Path("results/dec/vpcf/DEC_detailed_cluster_report.csv")
    elif model.lower() == "idec":
        results_file = Path("results/idec/vpcf/IDEC_detailed_cluster_report.csv")
    else:
        print(f"Unknown model: {model}")
        return
    
    if not h5_file.exists():
        print(f"Error: {h5_file} not found")
        return
    
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        print("Run the training pipeline first to generate cluster results.")
        return
    
    if output_dir is None:
        output_dir = f"results/{model}_visual_analysis"
    
    print(f"\nExporting {model.upper()} cluster analysis...")
    print(f"  H5 file: {h5_file}")
    print(f"  Results: {results_file}")
    print(f"  Output: {output_dir}")
    print(f"  Max samples per cluster: {max_samples}\n")
    
    create_cluster_visual_analysis(
        h5_filepath=h5_file,
        inspection_csv=results_file,
        output_dir=output_dir,
        source_name=model.lower(),
        max_samples_per_cluster=max_samples
    )
    
    print(f"\n✓ Analysis complete! Check {output_dir} for results.")


def sample_info_command(sample_id: int = 0):
    """Show info for a specific sample."""
    from vpcf_cluster_visual_mapping import VPCFClusterMapper
    
    h5_file = Path("data/vPCF_test_2.h5")
    
    if not h5_file.exists():
        print(f"Error: {h5_file} not found")
        return
    
    mapper = VPCFClusterMapper(h5_file, verbose=False)
    
    # Load both models if available
    dec_file = Path("results/dec/vpcf/DEC_detailed_cluster_report.csv")
    idec_file = Path("results/idec/vpcf/IDEC_detailed_cluster_report.csv")
    
    if dec_file.exists():
        mapper.load_inspection_results(dec_file, source_name="dec")
    if idec_file.exists():
        mapper.load_inspection_results(idec_file, source_name="idec")
    
    # Get sample info
    vpcf_data = mapper.get_vpcf_by_sample_id(sample_id, return_metadata=True)
    
    print(f"\nSample {sample_id} Information:")
    print(f"{'='*50}")
    print(f"  vPCF shape: {vpcf_data['vpcf_image'].shape}")
    
    if 'atomic_positions' in vpcf_data:
        pos = vpcf_data['atomic_positions']
        print(f"  Atomic position: ({pos[0]:.2f}, {pos[1]:.2f})")
    
    if 'vpcf_origin' in vpcf_data:
        origin = vpcf_data['vpcf_origin']
        print(f"  vPCF origin: ({origin[0]:.2f}, {origin[1]:.2f})")
    
    # Show cluster assignments
    if "dec" in mapper.inspection_results:
        dec_cluster = mapper.inspection_results["dec"][
            mapper.inspection_results["dec"]['sample_idx'] == sample_id
        ]['predicted_cluster'].values
        if len(dec_cluster) > 0:
            print(f"  DEC cluster: {dec_cluster[0]}")
    
    if "idec" in mapper.inspection_results:
        idec_cluster = mapper.inspection_results["idec"][
            mapper.inspection_results["idec"]['sample_idx'] == sample_id
        ]['predicted_cluster'].values
        if len(idec_cluster) > 0:
            print(f"  IDEC cluster: {idec_cluster[0]}")
    
    print()


def compare_command(output_dir: Optional[str] = None):
    """Compare DEC and IDEC results."""
    from vpcf_visual_analysis_integration import compare_dec_vs_idec_clusters
    
    h5_file = Path("data/vPCF_test_2.h5")
    dec_file = Path("results/dec/vpcf/DEC_detailed_cluster_report.csv")
    idec_file = Path("results/idec/vpcf/IDEC_detailed_cluster_report.csv")
    
    if not h5_file.exists():
        print(f"Error: {h5_file} not found")
        return
    
    if not dec_file.exists():
        print(f"Error: {dec_file} not found")
        return
    
    if not idec_file.exists():
        print(f"Error: {idec_file} not found")
        return
    
    if output_dir is None:
        output_dir = "results/dec_vs_idec_comparison"
    
    print(f"\nComparing DEC and IDEC results...")
    comparison_df = compare_dec_vs_idec_clusters(
        h5_filepath=h5_file,
        dec_results_path=dec_file,
        idec_results_path=idec_file,
        output_dir=output_dir
    )
    
    print(f"\n✓ Comparison complete! Check {output_dir} for results.")


def integrated_command(output_dir: Optional[str] = None):
    """Run complete integrated analysis."""
    from vpcf_visual_analysis_integration import create_integrated_visual_analysis
    
    h5_file = Path("data/vPCF_test_2.h5")
    dec_file = Path("results/dec/vpcf/DEC_detailed_cluster_report.csv")
    idec_file = Path("results/idec/vpcf/IDEC_detailed_cluster_report.csv")
    
    if not h5_file.exists():
        print(f"Error: {h5_file} not found")
        return
    
    if output_dir is None:
        output_dir = "results/integrated_visual_analysis"
    
    print(f"\nRunning integrated visual analysis...")
    create_integrated_visual_analysis(
        h5_filepath=h5_file,
        dec_results_path=dec_file if dec_file.exists() else None,
        idec_results_path=idec_file if idec_file.exists() else None,
        output_base_dir=output_dir
    )
    
    print(f"\n✓ Analysis complete! Check {output_dir} for results.")


def list_clusters_command(model: str = "dec"):
    """List all clusters for a model."""
    from vpcf_cluster_visual_mapping import VPCFClusterMapper
    
    h5_file = Path("data/vPCF_test_2.h5")
    
    if model.lower() == "dec":
        results_file = Path("results/dec/vpcf/DEC_detailed_cluster_report.csv")
    elif model.lower() == "idec":
        results_file = Path("results/idec/vpcf/IDEC_detailed_cluster_report.csv")
    else:
        print(f"Unknown model: {model}")
        return
    
    if not h5_file.exists():
        print(f"Error: {h5_file} not found")
        return
    
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return
    
    mapper = VPCFClusterMapper(h5_file, verbose=False)
    mapper.load_inspection_results(results_file, source_name=model.lower())
    
    stats = mapper.get_summary_statistics(source=model.lower())
    
    print(f"\n{model.upper()} Cluster Summary:")
    print(f"{'='*50}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of clusters: {stats['n_clusters']}")
    print(f"\nCluster sizes:")
    
    for cluster_id in sorted(stats['cluster_sizes'].keys()):
        size = stats['cluster_sizes'][cluster_id]
        pct = (size / stats['total_samples']) * 100
        print(f"  Cluster {cluster_id}: {size:5d} samples ({pct:5.1f}%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Quick-start script for vPCF cluster visual mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --export-clusters dec --max-samples 5
  %(prog)s --export-clusters idec
  %(prog)s --compare
  %(prog)s --integrated
  %(prog)s --sample-info 0
  %(prog)s --list-clusters dec
        """
    )
    
    parser.add_argument(
        "--export-clusters",
        choices=["dec", "idec"],
        metavar="MODEL",
        help="Export all clusters for a model (dec or idec)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum samples per cluster to export (default: 5)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (optional)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare DEC and IDEC results"
    )
    
    parser.add_argument(
        "--integrated",
        action="store_true",
        help="Run complete integrated analysis"
    )
    
    parser.add_argument(
        "--sample-info",
        type=int,
        metavar="SAMPLE_ID",
        help="Show information for a specific sample"
    )
    
    parser.add_argument(
        "--list-clusters",
        choices=["dec", "idec"],
        metavar="MODEL",
        help="List all clusters for a model"
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.export_clusters, args.compare, args.integrated, 
                args.sample_info is not None, args.list_clusters]):
        parser.print_help()
        return
    
    # Execute requested command
    if args.export_clusters:
        export_clusters_command(
            model=args.export_clusters,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
    
    if args.compare:
        compare_command(output_dir=args.output_dir)
    
    if args.integrated:
        integrated_command(output_dir=args.output_dir)
    
    if args.sample_info is not None:
        sample_info_command(sample_id=args.sample_info)
    
    if args.list_clusters:
        list_clusters_command(model=args.list_clusters)


if __name__ == "__main__":
    main()
