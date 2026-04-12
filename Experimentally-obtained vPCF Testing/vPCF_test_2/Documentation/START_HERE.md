# vPCF Cluster Visual Mapping - START HERE

## What You Have

A complete system for mapping DEC/IDEC cluster results back to original vPCF images and STEM data.

**In one sentence:** Cluster ID + Sample ID → vPCF image + Atomic position + Optional STEM location

## 5-Minute Quick Start

### Option 1: Command Line (No coding required)

```bash
cd "Experimentally-obtained vPCF Testing"

# See all available commands
python quick_start_visual_mapping.py --help

# Export clusters from DEC as visual images
python quick_start_visual_mapping.py --export-clusters dec --max-samples 5

# Compare DEC and IDEC results
python quick_start_visual_mapping.py --compare

# Look up a specific sample
python quick_start_visual_mapping.py --sample-info 42
```

### Option 2: Run Interactive Example

```bash
cd "Experimentally-obtained vPCF Testing"

# This will step through the entire workflow and create sample outputs
python interactive_example.py
```

### Option 3: Use in Python Script

```python
from vpcf_cluster_visual_mapping import VPCFClusterMapper

# Initialize mapper
mapper = VPCFClusterMapper("data/vPCF_test_2.h5")

# Load your DEC/IDEC results
mapper.load_inspection_results("results/dec/vpcf/DEC_detailed_cluster_report.csv", source_name="dec")

# Export cluster 0 as images + metadata
mapper.export_cluster_for_visual_analysis(
    cluster_id=0,
    output_dir="results/visual_analysis/cluster_0",
    source="dec",
    max_samples=10
)
```

## File Overview

### Main Implementation Files

| File | Purpose |
|------|---------|
| `vpcf_cluster_visual_mapping.py` | **Core mapper class** - Maps sample IDs to vPCF images |
| `vpcf_visual_analysis_integration.py` | Integration tools - Compare models, batch processing |
| `quick_start_visual_mapping.py` | **Command-line interface** - Easiest entry point |
| `interactive_example.py` | **Runnable example** - Steps through entire workflow |
| `example_visual_mapping.py` | 7 different usage examples |

### Documentation Files

| File | Contents |
|------|----------|
| `IMPLEMENTATION_SUMMARY.md` | This file - overview and quick start |
| `VISUAL_MAPPING_GUIDE.md` | **Complete API reference** - All functions explained |
| `README.md` | Detailed usage guide with workflows |

## How It Works

```
Your Inspection Results CSV
(sample_idx, predicted_cluster)
        ↓
VPCFClusterMapper
        ↓
H5 File (vPCF_test_2.h5)
├─ vpcf_images[sample_idx]      → PNG/visualization
├─ atomic_positions[sample_idx] → Scan coordinates (x,y)
└─ vpcf_origin[sample_idx]      → Origin info
        ↓
Output: Visual analysis for each cluster
├─ PNG montages of vPCF patterns
├─ CSV with metadata (atomic positions)
└─ Mapping reports
```

## What You Get

### For Each Cluster:

1. **Visual Output (PNG)**
   - Grid of vPCF images from the cluster
   - Quick visual inspection of patterns

2. **Metadata (CSV)**
   - Sample IDs in cluster
   - Cluster assignment
   - vPCF image shapes
   - Atomic positions (scan coordinates)
   - vPCF origins

3. **Mapping Report (CSV)**
   - Complete sample → cluster mapping
   - Atomic positions for each sample
   - Ready for further analysis

## Before You Start

**Required:** Existing DEC/IDEC cluster results
- Must have trained DEC/IDEC models
- Must have inspection results CSV: `results/dec/vpcf/DEC_detailed_cluster_report.csv`

**Optional:** Visualization and STEM mapping
- `matplotlib` (for image montages)
- `hyperspy` or `ncempy` (for DM3 STEM mapping)

Install with:
```bash
pip install matplotlib hyperspy
```

## Common Tasks

### View a specific sample's vPCF

```bash
python quick_start_visual_mapping.py --sample-info 42
```

### List all clusters and their sizes

```bash
python quick_start_visual_mapping.py --list-clusters dec
```

### Create visual analysis for all clusters

```bash
python quick_start_visual_mapping.py --export-clusters dec --max-samples 5
```

### Compare DEC vs IDEC clustering

```bash
python quick_start_visual_mapping.py --compare
```

### Full integrated analysis

```bash
python quick_start_visual_mapping.py --integrated
```

## Python API (For Scripts)

```python
from vpcf_cluster_visual_mapping import VPCFClusterMapper

# 1. Initialize
mapper = VPCFClusterMapper("data/vPCF_test_2.h5")

# 2. Load results
mapper.load_inspection_results(
    "results/dec/vpcf/DEC_detailed_cluster_report.csv",
    source_name="dec"
)

# 3. Explore clusters
cluster_samples = mapper.get_cluster_samples(0, source="dec")  # Get samples in cluster 0
info = mapper.get_cluster_info(0, source="dec")               # Get cluster statistics

# 4. Get sample data
vpcf_data = mapper.get_vpcf_by_sample_id(0, return_metadata=True)
# Returns: {'vpcf_image': ndarray, 'atomic_positions': [x,y], 'vpcf_origin': [x,y]}

# 5. Export for analysis
mapper.export_cluster_for_visual_analysis(
    cluster_id=0,
    output_dir="results/visual_analysis/cluster_0",
    source="dec",
    max_samples=10
)
```

## Next Steps

1. **Verify Setup** (1 minute)
   ```bash
   cd "Experimentally-obtained vPCF Testing"
   python quick_start_visual_mapping.py --help
   ```

2. **Run Example** (2 minutes)
   ```bash
   python interactive_example.py
   ```

3. **Export Your Data** (5 minutes)
   ```bash
   python quick_start_visual_mapping.py --export-clusters dec --max-samples 5
   ```

4. **Review Output**
   - Check `results/visual_analysis/` for PNG montages
   - Check CSV files for metadata
   - Compare with your expectations

5. **Read Full Guide** (optional, for advanced usage)
   - `VISUAL_MAPPING_GUIDE.md` for complete API reference
   - `example_visual_mapping.py` for more examples

## Features at a Glance

| Feature | Command | Python |
|---------|---------|--------|
| Get vPCF image | `--sample-info 0` | `mapper.get_vpcf_by_sample_id(0)` |
| List clusters | `--list-clusters dec` | `mapper.get_summary_statistics()` |
| Export cluster | `--export-clusters dec` | `mapper.export_cluster_for_visual_analysis()` |
| Compare models | `--compare` | `compare_dec_vs_idec_clusters()` |
| Full analysis | `--integrated` | `create_integrated_visual_analysis()` |
| Get cluster samples | N/A | `mapper.get_cluster_samples(0)` |
| Create mapping report | N/A | `mapper.create_cluster_mapping_report()` |
| Map to STEM | N/A | `mapper.map_to_stem_if_available()` |

## FAQ

**Q: My inspection results CSV doesn't exist**
A: You need to generate it first. Train your DEC/IDEC models and they should create the CSV automatically.

**Q: Can I use this without matplotlib?**
A: Yes! CSV metadata will still be saved. PNG montages just won't be created.

**Q: What if I have a DM3 file? Can I map back to STEM?**
A: Yes! Use `mapper.map_to_stem_if_available(sample_id, dm3_filepath)`. The atomic positions from the H5 file are used as scan coordinates.

**Q: How many samples can I handle?**
A: Tested with 7205 samples. Should scale to 100k+ with batch processing.

**Q: Can I use this in a Jupyter notebook?**
A: Yes! Import the module and use it like the Python API examples above.

**Q: Where do the outputs go?**
A: By default, `results/visual_analysis/` but you can specify any directory.

## Troubleshooting

### "No inspection results loaded"
→ Load results first: `mapper.load_inspection_results(...)`

### "H5 file not found"
→ Check path is correct and file exists in `data/` folder

### "DEC_detailed_cluster_report.csv not found"
→ Run training pipeline first to generate the CSV

### matplotlib error when saving images
→ Install: `pip install matplotlib`

### Import error on hyperspy/ncempy
→ Optional - only needed for STEM mapping. Install one: `pip install hyperspy` or `pip install ncempy`

## Documentation

- **`VISUAL_MAPPING_GUIDE.md`** - Complete function reference and workflows
- **`IMPLEMENTATION_SUMMARY.md`** - Technical details and architecture
- **`example_visual_mapping.py`** - 7 complete usage examples
- **`interactive_example.py`** - Step-by-step walkthrough

## Support

1. Check **VISUAL_MAPPING_GUIDE.md** for API questions
2. Run **interactive_example.py** to see how it works
3. Check **example_visual_mapping.py** for different scenarios
4. Review docstrings in the Python modules (all functions fully documented)

---

**Ready to start?** Run this command now:
```bash
cd "Experimentally-obtained vPCF Testing"
python interactive_example.py
```

Then check the `results/` directory for generated outputs!
