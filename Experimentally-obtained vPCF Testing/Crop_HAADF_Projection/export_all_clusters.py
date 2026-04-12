"""
Export cluster coordinates for both DEC and IDEC methods for HAADF data.
"""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from vpcf_to_image_mapper import VPCFToImageMapper

workspace_root = Path(__file__).parent
h5_file = workspace_root / "data" / "Crop_HAADF_Projection_80pixels.h5"

# Export for DEC
print("\n" + "="*60)
print("Processing DEC clustering results...")
print("="*60)
dec_dir = workspace_root / "results" / "h5_only" / "dec"
dec_mapper = VPCFToImageMapper(
    h5_file=str(h5_file),
    clustering_results_dir=str(dec_dir),
    method='dec'
)
dec_mapper.summary()

dec_output = workspace_root / "results" / "dec_cluster_coordinates.csv"
dec_df = dec_mapper.export_cluster_coordinates(str(dec_output))
print(f"✓ DEC results exported: {dec_output}")

# Export for IDEC
print("\n" + "="*60)
print("Processing IDEC clustering results...")
print("="*60)
idec_dir = workspace_root / "results" / "h5_only" / "idec"
idec_mapper = VPCFToImageMapper(
    h5_file=str(h5_file),
    clustering_results_dir=str(idec_dir),
    method='idec'
)
idec_mapper.summary()

idec_output = workspace_root / "results" / "idec_cluster_coordinates.csv"
idec_df = idec_mapper.export_cluster_coordinates(str(idec_output))
print(f"✓ IDEC results exported: {idec_output}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"DEC:  {dec_output}")
print(f"IDEC: {idec_output}")
print("="*60 + "\n")
