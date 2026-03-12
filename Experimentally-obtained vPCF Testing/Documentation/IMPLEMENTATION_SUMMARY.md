# vPCF Cluster Visual Mapping - Implementation Summary

## What Was Created

I've built a comprehensive system for mapping DEC/IDEC cluster results back to original vPCF images and STEM data for visual analysis. Here's what's included:

### Core Files

1. **`vpcf_cluster_visual_mapping.py`** (Main Module)
   - `VPCFClusterMapper` class: Maps sample IDs + clusters → vPCF images + metadata
   - Key methods:
     - `load_inspection_results()` - Load DEC/IDEC cluster CSV files
     - `get_vpcf_by_sample_id()` - Retrieve vPCF image and metadata
     - `get_cluster_samples()` - Get all samples in a cluster
     - `export_cluster_for_visual_analysis()` - Create visual reports
     - `map_to_stem_if_available()` - Map back to STEM images (if DM3 available)
   - Tested and verified working with your H5 file (7205 samples)

2. **`vpcf_visual_analysis_integration.py`** (Integration Tools)
   - `compare_dec_vs_idec_clusters()` - Compare model results side-by-side
   - `create_integrated_visual_analysis()` - One-click batch processing
   - `generate_inspection_results_for_trained_model()` - Create CSVs from trained models

3. **`quick_start_visual_mapping.py`** (Command-Line Interface)
   - Easy-to-use CLI for common tasks:
     ```bash
     python quick_start_visual_mapping.py --export-clusters dec --max-samples 5
     python quick_start_visual_mapping.py --compare
     python quick_start_visual_mapping.py --sample-info 42
     ```

4. **`example_visual_mapping.py`** (Usage Examples)
   - 7 complete example workflows showing how to use the mapper

5. **`VISUAL_MAPPING_GUIDE.md`** (Complete Documentation)
   - Full API reference, workflows, troubleshooting, and advanced usage

## Quick Start (30 seconds)

### Option 1: Command Line (Easiest)

```bash
cd "Experimentally-obtained vPCF Testing"

# Export all clusters from DEC
python quick_start_visual_mapping.py --export-clusters dec --max-samples 5

# Compare DEC vs IDEC
python quick_start_visual_mapping.py --compare

# See info about specific sample
python quick_start_visual_mapping.py --sample-info 0
```

### Option 2: Python Script

```python
from vpcf_cluster_visual_mapping import VPCFClusterMapper

# Initialize mapper
mapper = VPCFClusterMapper("data/vPCF_test_2.h5")

# Load DEC results
mapper.load_inspection_results(
    "results/dec/vpcf/DEC_detailed_cluster_report.csv",
    source_name="dec"
)

# Export cluster 0 for visual analysis
mapper.export_cluster_for_visual_analysis(
    cluster_id=0,
    output_dir="results/visual_analysis/cluster_0",
    source="dec",
    max_samples=10
)

# Get summary
print(mapper.get_summary_statistics(source="dec"))
```

## How It Works

```
Your DEC/IDEC Inspection Results
    ↓
Sample ID → Cluster assignment
    ↓
VPCFClusterMapper.load_inspection_results()
    ↓
    ├→ Sample index (0-7204)
    ├→ Cluster ID (0-K)
    └→ Match against H5 file
        ↓
        H5 File (vPCF_test_2.h5)
        ├→ vpcf_images[sample_idx] 
        ├→ atomic_positions[sample_idx] (scan coords)
        └→ vpcf_origin[sample_idx]
        ↓
    Visual Output
    ├→ vPCF image montages (PNG)
    ├→ Metadata CSV with atomic positions
    └→ Sample-to-cluster mapping reports
```

## Data Available Per Sample

When you retrieve a sample using `mapper.get_vpcf_by_sample_id(0, return_metadata=True)`, you get:

```python
{
    'vpcf_image': np.ndarray of shape (999, 999),  # The vPCF pattern
    'atomic_positions': array([1623., 1344.]),      # Scan position (x, y)
    'vpcf_origin': array([500., 498.])             # vPCF origin point
}
```

The atomic positions correspond to scan coordinates in the STEM image, allowing you to map back to the original experimental STEM data if a DM3 file is available.

## Output Structure

Running `export_cluster_for_visual_analysis()` creates:

```
cluster_0/
├── cluster_0_vpcf_montage_dec.png    # Visual grid of vPCF images
└── cluster_0_samples_metadata.csv    # Metadata with atomic positions
    Columns: index_in_cluster, sample_id, cluster_id, vpcf_shape,
             atomic_x, atomic_y, origin_x, origin_y
```

## Example: Map a Specific Sample

```python
from vpcf_cluster_visual_mapping import VPCFClusterMapper

mapper = VPCFClusterMapper("data/vPCF_test_2.h5")

# Get vPCF for sample 42
vpcf_data = mapper.get_vpcf_by_sample_id(42, return_metadata=True)

# Extract information
vpcf_image = vpcf_data['vpcf_image']              # Shape: (999, 999)
scan_position = vpcf_data['atomic_positions']    # [x, y] in STEM image
origin = vpcf_data['vpcf_origin']                # [x, y] in vPCF

# Now you can:
# 1. Visualize the vPCF image
# 2. Use scan_position to locate in STEM image (if DM3 loaded)
# 3. Correlate with clustering results
```

## Key Features

✓ **Sample-to-Cluster Mapping**
  - Load DEC/IDEC inspection results
  - Get all samples in a cluster
  - Retrieve vPCF images by sample ID

✓ **Visual Analysis Tools**
  - Generate PNG montages of cluster vPCF patterns
  - Create metadata CSVs with atomic positions
  - Export detailed mapping reports

✓ **STEM Integration**
  - Extract atomic positions (scan coordinates)
  - Optionally map back to original DM3 STEM images
  - Correlate vPCF clusters with STEM regions

✓ **Model Comparison**
  - Compare DEC and IDEC cluster assignments
  - Identify samples where models disagree
  - Export disagreement analysis

✓ **Batch Processing**
  - Process all clusters at once
  - Create analysis for all models in one command
  - Scalable to large datasets (7205+ samples)

## Requirements

### Essential
- `h5py` - Read H5 files
- `numpy` - Array operations
- `pandas` - Data handling

### Optional
- `matplotlib` - For image montage visualization
- `hyperspy` or `ncempy` - For STEM image mapping to DM3 files

Install with:
```bash
pip install h5py numpy pandas matplotlib hyperspy
```

## Tested With Your Data

The module has been tested and verified with your actual setup:
- **H5 file:** 7205 samples, each with 999×999 vPCF images
- **Metadata:** Atomic positions and vPCF origin data verified in H5
- **Use case:** Maps cluster results back to original experimental data

## Next Steps

1. **Run Quick Test:**
   ```bash
   cd "Experimentally-obtained vPCF Testing"
   python quick_start_visual_mapping.py --help
   ```

2. **Export Your Clusters:**
   ```bash
   # This will create visual analysis for all your clusters
   python quick_start_visual_mapping.py --export-clusters dec --max-samples 5
   ```

3. **Compare Models:**
   ```bash
   python quick_start_visual_mapping.py --compare
   ```

4. **Read Full Guide:** See `VISUAL_MAPPING_GUIDE.md` for detailed documentation

## File Locations

All new files are in: `Experimentally-obtained vPCF Testing/`
- `vpcf_cluster_visual_mapping.py` - Main mapper module
- `vpcf_visual_analysis_integration.py` - Integration utilities
- `quick_start_visual_mapping.py` - CLI interface
- `example_visual_mapping.py` - Usage examples
- `VISUAL_MAPPING_GUIDE.md` - Complete documentation
- `IMPLEMENTATION_SUMMARY.md` - This file

## Architecture

The mapper uses a **non-invasive design** that:
- Doesn't modify your existing training code
- Works with inspection results CSV files
- Creates new output directories (doesn't overwrite existing data)
- Can be integrated into your existing pipeline
- Supports both DEC and IDEC models

## Common Use Cases Supported

1. **Visual Inspection of Clusters**
   - Generate PNG montages of vPCF patterns per cluster
   - Identify similar patterns across samples

2. **Sample Exploration**
   - Look up which cluster a specific sample belongs to
   - View its vPCF and atomic position info

3. **Model Comparison**
   - See where DEC and IDEC agree/disagree
   - Export disagreement analysis for investigation

4. **STEM Correlation**
   - Map vPCF samples back to scan positions
   - Correlate clusters with STEM regions

5. **Quality Analysis**
   - Review cluster compositions
   - Check for outliers or anomalies visually

## Troubleshooting

**Q: "DEC_detailed_cluster_report.csv not found"**
A: The inspection results CSV doesn't exist. You need to either:
   - Run the training pipeline to generate it
   - Or manually create inspection results with `create_cluster_inspection_report()` from `train_vpcf_models.py`

**Q: matplotlib not available?**
A: Image montages won't be saved. Install with: `pip install matplotlib`

**Q: How do I use this with my trained models?**
A: See `vpcf_visual_analysis_integration.generate_inspection_results_for_trained_model()` for generating inspection CSVs from model weights.

## Support

For questions or issues:
1. Check `VISUAL_MAPPING_GUIDE.md` - API reference section
2. Review examples in `example_visual_mapping.py`
3. Check the integration guide in `vpcf_visual_analysis_integration.py`

---

**Status:** Complete and tested with your data
**Last Updated:** 2026-02-26
