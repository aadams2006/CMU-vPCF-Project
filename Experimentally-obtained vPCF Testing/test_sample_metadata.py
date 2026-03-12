"""
Test and verify the sample metadata lookup system.

This script tests all three components:
1. h5_metadata_inspector - diagnostic of H5 file structure
2. sample_metadata_lookup.SampleMetadataManager - unified metadata access
3. vpcf_cluster_visual_mapping.VPCFClusterMapper extended methods - retrieve sample metadata

Usage:
    python test_sample_metadata.py
    or
    python test_sample_metadata.py --full  (runs all tests)
    or
    python test_sample_metadata.py --sample 123  (inspect sample 123)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure we can import local modules
sys.path.insert(0, str(Path(__file__).parent))


def test_h5_inspector():
    """Test the H5 metadata inspector."""
    print("\n" + "=" * 70)
    print("TEST 1: H5 Metadata Inspector")
    print("=" * 70)
    
    from h5_metadata_inspector import inspect_h5_structure, inspect_sample_metadata
    
    h5_file = Path("data/vPCF_test_2.h5")
    if not h5_file.exists():
        print(f"ERROR: H5 file not found at {h5_file}")
        return False
    
    print(f"\nInspecting H5 file: {h5_file}")
    inspect_h5_structure(h5_file)
    
    # Inspect a few samples
    print("\n[Inspecting specific samples]")
    inspect_sample_metadata(h5_file, [0, 1, 2])
    
    return True


def test_vpcf_mapper_metadata():
    """Test the extended VPCFClusterMapper metadata methods."""
    print("\n" + "=" * 70)
    print("TEST 2: VPCFClusterMapper Extended Methods")
    print("=" * 70)
    
    from vpcf_cluster_visual_mapping import VPCFClusterMapper
    
    h5_file = Path("data/vPCF_test_2.h5")
    if not h5_file.exists():
        print(f"ERROR: H5 file not found at {h5_file}")
        return False
    
    print(f"\nInitializing VPCFClusterMapper...")
    mapper = VPCFClusterMapper(h5_file, verbose=True)
    
    print("\n[H5 File Structure]")
    mapper.print_h5_structure()
    
    print("\n[Getting sample metadata for samples 0, 123, 500]")
    for sample_id in [0, 123, 500]:
        try:
            metadata = mapper.get_sample_metadata(sample_id, include_vpcf_image=False)
            print(f"\nSample {sample_id}:")
            for key, val in metadata.items():
                print(f"  {key}: {val}")
        except Exception as e:
            print(f"ERROR getting sample {sample_id}: {e}")
            return False
    
    return True


def test_sample_metadata_manager():
    """Test the SampleMetadataManager."""
    print("\n" + "=" * 70)
    print("TEST 3: SampleMetadataManager")
    print("=" * 70)
    
    from sample_metadata_lookup import SampleMetadataManager
    
    h5_file = Path("data/vPCF_test_2.h5")
    if not h5_file.exists():
        print(f"ERROR: H5 file not found at {h5_file}")
        return False
    
    print(f"\nInitializing SampleMetadataManager...")
    manager = SampleMetadataManager(h5_file, verbose=True)
    
    print("\n[Summary Report]")
    manager.print_summary()
    
    print("\n[Getting sample info for sample 123]")
    try:
        info = manager.get_sample_info(123, include_image=False)
        print(f"Sample 123 info:")
        for key, val in info.items():
            if key != 'vpcf_image':
                print(f"  {key}: {val}")
    except Exception as e:
        print(f"ERROR getting sample 123: {e}")
        # This might fail if no labels are available - that's OK
    
    # Try to search by crystal type if labels are available
    try:
        print("\n[Searching by crystal type]")
        if manager.labels_df is not None:
            crystal_types = manager.get_crystal_types()
            print(f"Found {len(crystal_types)} crystal types:")
            for ct, samples in sorted(crystal_types.items())[:5]:
                print(f"  {ct}: {len(samples)} samples")
        else:
            print("  (No external label file loaded, skipping crystal type search)")
    except Exception as e:
        print(f"Note: Crystal type search not available: {e}")
    
    # Test export
    print("\n[Testing metadata export]")
    try:
        output_file = Path("test_export_metadata.csv")
        result = manager.export_sample_metadata(
            sample_ids=[0, 1, 2, 3, 4],
            output_file=output_file,
            include_images=False
        )
        print(f"Export result: {result}")
        if output_file.exists():
            print(f"✓ Export file created: {output_file}")
            # Read and display
            import pandas as pd
            df = pd.read_csv(output_file)
            print("First few rows:")
            print(df.head())
            # Clean up
            output_file.unlink()
            print("(Temporary file deleted)")
        else:
            print(f"WARNING: Export file not created at {output_file}")
    except Exception as e:
        print(f"ERROR during export: {e}")
        return False
    
    return True


def test_unified_workflow():
    """Test a complete unified workflow."""
    print("\n" + "=" * 70)
    print("TEST 4: Unified Workflow - Sample 123")
    print("=" * 70)
    
    from vpcf_cluster_visual_mapping import VPCFClusterMapper
    from sample_metadata_lookup import SampleMetadataManager
    
    h5_file = Path("data/vPCF_test_2.h5")
    sample_id = 123
    
    print(f"\n[Using both mapper and manager for sample {sample_id}]")
    
    # Method 1: Using VPCFClusterMapper
    print("\nMethod 1: VPCFClusterMapper.get_sample_metadata()")
    try:
        mapper = VPCFClusterMapper(h5_file, verbose=False)
        mapper_data = mapper.get_sample_metadata(sample_id, include_vpcf_image=False)
        print(f"  atomic_positions: {mapper_data.get('atomic_positions')}")
        print(f"  vpcf_origin: {mapper_data.get('vpcf_origin')}")
        print(f"  vpcf_shape: {mapper_data.get('vpcf_shape')}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    # Method 2: Using SampleMetadataManager
    print("\nMethod 2: SampleMetadataManager.get_sample_info()")
    try:
        manager = SampleMetadataManager(h5_file, verbose=False)
        manager_data = manager.get_sample_info(sample_id, include_image=False)
        print(f"  sample_id: {manager_data.get('sample_id')}")
        print(f"  atomic_positions: {manager_data.get('atomic_positions')}")
        print(f"  vpcf_origin: {manager_data.get('vpcf_origin')}")
        print(f"  vpcf_shape: {manager_data.get('vpcf_shape')}")
        
        if 'crystal_label' in manager_data:
            print(f"  crystal_label: {manager_data.get('crystal_label')}")
            print(f"  crystal_type: {manager_data.get('crystal_type')}")
            print(f"  orientation_vector: {manager_data.get('orientation_vector')}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    print("\n✓ Unified workflow test passed!")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test sample metadata lookup system"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Inspect a specific sample"
    )
    parser.add_argument(
        "--test-inspector",
        action="store_true",
        help="Test H5 metadata inspector only"
    )
    parser.add_argument(
        "--test-mapper",
        action="store_true",
        help="Test VPCFClusterMapper extended methods only"
    )
    parser.add_argument(
        "--test-manager",
        action="store_true",
        help="Test SampleMetadataManager only"
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    results = {}
    
    # Determine which tests to run
    run_all = args.full or (not args.test_inspector and not args.test_mapper and 
                             not args.test_manager and args.sample is None)
    
    if run_all or args.test_inspector:
        print("\n[Running H5 Inspector Test]")
        results['H5 Inspector'] = test_h5_inspector()
    
    if run_all or args.test_mapper:
        print("\n[Running VPCFClusterMapper Test]")
        results['VPCFClusterMapper'] = test_vpcf_mapper_metadata()
    
    if run_all or args.test_manager:
        print("\n[Running SampleMetadataManager Test]")
        results['SampleMetadataManager'] = test_sample_metadata_manager()
    
    if run_all:
        print("\n[Running Unified Workflow Test]")
        results['Unified Workflow'] = test_unified_workflow()
    
    # Test specific sample if requested
    if args.sample is not None:
        print("\n" + "=" * 70)
        print(f"CUSTOM TEST: Inspect Sample {args.sample}")
        print("=" * 70)
        
        from sample_metadata_lookup import SampleMetadataManager
        
        try:
            manager = SampleMetadataManager("data/vPCF_test_2.h5", verbose=True)
            info = manager.get_sample_info(args.sample, include_image=False)
            print(f"\n✓ Sample {args.sample} Info:")
            for key, val in info.items():
                if key != 'vpcf_image':
                    print(f"  {key}: {val}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[f'Sample {args.sample}'] = False
        else:
            results[f'Sample {args.sample}'] = True
    
    # Print summary
    if results:
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name}: {status}")
        
        all_passed = all(results.values())
        print("\n" + ("=" * 70))
        if all_passed:
            print("ALL TESTS PASSED ✓")
        else:
            print("SOME TESTS FAILED ✗")
        print("=" * 70)
        
        return 0 if all_passed else 1
    
    return 0


if __name__ == "__main__":
    import os
    sys.exit(main())
