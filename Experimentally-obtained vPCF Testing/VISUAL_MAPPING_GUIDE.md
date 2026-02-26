# vPCF Cluster Visual Mapping - Complete Guide

## Overview

This guide explains how to map DEC/IDEC cluster assignments back to original vPCF images and their corresponding STEM data for visual analysis.

**What it does:**
- Maps cluster IDs → sample indices from inspection results
- Retrieves vPCF images from the H5 file for visual inspection
- Extracts atomic positions (scan coordinates) from metadata
- Creates visual montages of vPCF images grouped by cluster
- Optionally maps back to STEM images if DM3 files are available
- Generates detailed CSV reports with metadata for each cluster

## Files Created

### Core Modules

1. **`vpcf_cluster_visual_mapping.py`** - Main mapping module
   - `VPCFClusterMapper` class: Core functionality for mapping clusters to vPCF data
   - `create_cluster_visual_analysis()`: Convenience function for batch processing

2. **`vpcf_visual_analysis_integration.py`** - Integration utilities
   - `compare_dec_vs_idec_clusters()`: Compare DEC and IDEC results side-by-side
   - `create_integrated_visual_analysis()`: One-stop analysis of all clusters
   - `generate_inspection_results_for_trained_model()`: Generate inspection CSVs from trained models

3. **`example_visual_mapping.py`** - Example usage with 7 different scenarios

## Quick Start

### 1. Basic Usage - Get vPCF for a Sample

```python
from vpcf_cluster_visual_mapping import VPCFClusterMapper
from pathlib import Path

# Initialize mapper
mapper = VPCFClusterMapper("data/vPCF_test_2.h5")

# Retrieve vPCF image and metadata for sample 0
vpcf_data = mapper.get_vpcf_by_sample_id(sample_id=0, return_metadata=True)

print(f"vPCF shape: {vpcf_data['vpcf_image'].shape}")
print(f"Atomic position: {vpcf_data['atomic_positions']}")
```

### 2. Load Inspection Results

```python
# Load DEC results
dec_results = mapper.load_inspection_results(
    "results/dec/vpcf/DEC_detailed_cluster_report.csv",
    source_name="dec"
)

# Load IDEC results
idec_results = mapper.load_inspection_results(
    "results/idec/vpcf/IDEC_detailed_cluster_report.csv",
    source_name="idec"
)

# Get summary statistics
print(mapper.get_summary_statistics(source="dec"))
```

### 3. Export Cluster for Visual Analysis

```python
# Export cluster 0 with up to 10 samples
cluster_info = mapper.export_cluster_for_visual_analysis(
    cluster_id=0,
    output_dir="results/visual_analysis/cluster_0",
    source="dec",
    max_samples=10,
    save_images=True,  # Creates PNG montage
    save_csv=True      # Creates metadata CSV
)
```

### 4. Batch Process All Clusters

```python
from vpcf_cluster_visual_mapping import create_cluster_visual_analysis

# Create analysis for all clusters
create_cluster_visual_analysis(
    h5_filepath="data/vPCF_test_2.h5",
    inspection_csv="results/dec/vpcf/DEC_detailed_cluster_report.csv",
    output_dir="results/dec_visual_analysis",
    source_name="dec",
    max_samples_per_cluster=5
)
```

### 5. Compare DEC vs IDEC

```python
from vpcf_visual_analysis_integration import compare_dec_vs_idec_clusters

comparison = compare_dec_vs_idec_clusters(
    h5_filepath="data/vPCF_test_2.h5",
    dec_results_path="results/dec/vpcf/DEC_detailed_cluster_report.csv",
    idec_results_path="results/idec/vpcf/IDEC_detailed_cluster_report.csv",
    output_dir="results/comparison"
)

# See which samples have different cluster assignments
print(comparison[~comparison['clusters_match']])
```

### 6. Complete Integrated Analysis

```python
from vpcf_visual_analysis_integration import create_integrated_visual_analysis

create_integrated_visual_analysis(
    h5_filepath="data/vPCF_test_2.h5",
    dec_results_path="results/dec/vpcf/DEC_detailed_cluster_report.csv",
    idec_results_path="results/idec/vpcf/IDEC_detailed_cluster_report.csv",
    output_base_dir="results/integrated_analysis"
)
```

## Data Flow

```
DEC/IDEC Inspection Results CSV
├── sample_idx (0 to N-1)
├── predicted_cluster (0 to K-1)
└── (optional) ground_truth_cluster
        ↓
VPCFClusterMapper
        ↓
H5 File (vPCF_test_2.h5)
├── vpcf_images[sample_idx] → vPCF image to visualize
├── atomic_positions[sample_idx] → scan coordinates
└── vpcf_origin[sample_idx] → origin info
        ↓
Visual Analysis Output
├── cluster_N_vpcf_montage.png (vPCF images for cluster N)
├── cluster_N_samples_metadata.csv (atomic positions, origins)
└── sample_to_cluster_mapping.csv (complete mapping with metadata)
```

## Output Structure

When you run `export_cluster_for_visual_analysis()` or `create_cluster_visual_analysis()`, you get:

```
output_dir/
├── cluster_0/
│   ├── cluster_0_vpcf_montage_dec.png
│   ├── cluster_0_samples_metadata.csv
│   └── sample_to_cluster_mapping.csv
├── cluster_1/
│   ├── cluster_1_vpcf_montage_dec.png
│   ├── cluster_1_samples_metadata.csv
│   └── sample_to_cluster_mapping.csv
└── ...
```

### Sample Metadata CSV

```
index_in_cluster,sample_id,cluster_id,vpcf_shape,atomic_x,atomic_y,origin_x,origin_y
0,0,0,"(999, 999)",123.45,234.56,500.0,500.0
1,15,0,"(999, 999)",124.12,235.23,500.0,500.0
...
```

## API Reference

### VPCFClusterMapper Class

#### Initialization
```python
mapper = VPCFClusterMapper(h5_filepath, verbose=True)
```

#### Loading Results
```python
df = mapper.load_inspection_results(csv_filepath, source_name="dec")
```

#### Retrieving Data
```python
# Get vPCF for single sample
vpcf_data = mapper.get_vpcf_by_sample_id(sample_id=0, return_metadata=True)

# Get all samples in a cluster
sample_ids = mapper.get_cluster_samples(cluster_id=0, source="dec")

# Get cluster information
info = mapper.get_cluster_info(cluster_id=0, source="dec")
```

#### Creating Reports
```python
# Single cluster export
mapper.export_cluster_for_visual_analysis(
    cluster_id=0,
    output_dir="path/to/output",
    source="dec",
    max_samples=None,
    save_images=True,
    save_csv=True
)

# Create mapping report with atomic positions
report_df = mapper.create_cluster_mapping_report(
    source="dec",
    output_dir="results/analysis"
)

# Get summary statistics
stats = mapper.get_summary_statistics(source="dec")
```

#### Mapping to STEM
```python
# Map vPCF sample to STEM image (if DM3 available)
mapping = mapper.map_to_stem_if_available(
    sample_id=0,
    dm3_filepath="data/vPCF_test_2.dm3"
)
```

### Convenience Functions

```python
from vpcf_cluster_visual_mapping import create_cluster_visual_analysis

create_cluster_visual_analysis(
    h5_filepath="data/vPCF_test_2.h5",
    inspection_csv="results/dec/vpcf/DEC_detailed_cluster_report.csv",
    output_dir="results/dec_visual_analysis",
    source_name="dec",
    max_samples_per_cluster=10
)
```

## Common Workflows

### Workflow 1: Analyze DEC Results Only

```python
from vpcf_cluster_visual_mapping import VPCFClusterMapper

mapper = VPCFClusterMapper("data/vPCF_test_2.h5")
mapper.load_inspection_results(
    "results/dec/vpcf/DEC_detailed_cluster_report.csv",
    source_name="dec"
)

# Export all clusters for visual inspection
for cluster_id in range(mapper.get_summary_statistics("dec")['n_clusters']):
    mapper.export_cluster_for_visual_analysis(
        cluster_id=cluster_id,
        output_dir=f"results/dec_clusters/cluster_{cluster_id}",
        source="dec",
        max_samples=5
    )
```

### Workflow 2: Compare Model Results

```python
from vpcf_visual_analysis_integration import compare_dec_vs_idec_clusters

# Compare which samples get assigned to different clusters
comparison_df = compare_dec_vs_idec_clusters(
    h5_filepath="data/vPCF_test_2.h5",
    dec_results_path="results/dec/vpcf/DEC_detailed_cluster_report.csv",
    idec_results_path="results/idec/vpcf/IDEC_detailed_cluster_report.csv",
    output_dir="results/model_comparison"
)

# Look at disagreements
disagreements = comparison_df[~comparison_df['clusters_match']]
print(f"Models disagree on {len(disagreements)} samples ({len(disagreements)/len(comparison_df)*100:.1f}%)")
```

### Workflow 3: Interactive Exploration (Jupyter)

```python
from vpcf_cluster_visual_mapping import VPCFClusterMapper
import matplotlib.pyplot as plt

mapper = VPCFClusterMapper("data/vPCF_test_2.h5")
mapper.load_inspection_results(
    "results/dec/vpcf/DEC_detailed_cluster_report.csv",
    source_name="dec"
)

# Explore a specific sample
sample_id = 42
vpcf_data = mapper.get_vpcf_by_sample_id(sample_id)

plt.figure(figsize=(10, 8))
plt.imshow(vpcf_data['vpcf_image'], cmap='viridis')
plt.title(f"Sample {sample_id}")
plt.colorbar()
plt.show()

# Get cluster info for this sample
cluster_of_sample = mapper.inspection_results['dec'][
    mapper.inspection_results['dec']['sample_idx'] == sample_id
]['predicted_cluster'].values[0]

cluster_info = mapper.get_cluster_info(cluster_of_sample, source="dec")
print(f"Sample {sample_id} is in cluster {cluster_of_sample}")
print(f"This cluster has {cluster_info['sample_count']} samples")
```

## Requirements

### Core Dependencies
- `numpy` - Array operations
- `pandas` - DataFrames for CSV handling
- `h5py` - Reading H5 files

### Optional Dependencies
- `matplotlib` - For image montage visualization
- `hyperspy` or `ncempy` - For DM3 STEM image loading (optional for STEM mapping)

### Install Dependencies
```bash
pip install h5py numpy pandas matplotlib hyperspy
# or for ncempy instead of hyperspy:
pip install h5py numpy pandas matplotlib ncempy
```

## Troubleshooting

### "No inspection results loaded"
**Problem:** Error when trying to export clusters without loading inspection results first.

**Solution:** Load inspection results first:
```python
mapper.load_inspection_results(
    "results/dec/vpcf/DEC_detailed_cluster_report.csv",
    source_name="dec"
)
```

### "DEC_detailed_cluster_report.csv not found"
**Problem:** Inspection results CSV doesn't exist in results directory.

**Solution:** 
1. Ensure DEC/IDEC training has completed
2. Training pipeline should generate this file automatically
3. You can also create it using `vpcf_visual_analysis_integration.generate_inspection_results_for_trained_model()`

### matplotlib not available for image saving
**Problem:** Images won't save when `save_images=True`.

**Solution:** Install matplotlib:
```bash
pip install matplotlib
```

### "Neither hyperspy nor ncempy available" (when mapping to STEM)
**Problem:** Cannot load DM3 files for STEM image mapping.

**Solution:** Install one of the dependencies:
```bash
pip install hyperspy  # or
pip install ncempy
```

## Advanced Usage

### Custom Cluster Analysis

```python
from vpcf_cluster_visual_mapping import VPCFClusterMapper
import numpy as np

mapper = VPCFClusterMapper("data/vPCF_test_2.h5")
mapper.load_inspection_results("results/dec/vpcf/DEC_detailed_cluster_report.csv")

# Get cluster 0
cluster_0_samples = mapper.get_cluster_samples(0, source="dec")

# Load all vPCF images for the cluster
vpcf_images = []
atomic_positions = []

for sample_id in cluster_0_samples[:10]:
    data = mapper.get_vpcf_by_sample_id(sample_id, return_metadata=True)
    vpcf_images.append(data['vpcf_image'])
    if 'atomic_positions' in data:
        atomic_positions.append(data['atomic_positions'])

vpcf_images = np.array(vpcf_images)
atomic_positions = np.array(atomic_positions)

print(f"Loaded {len(vpcf_images)} vPCF images from cluster 0")
print(f"Mean vPCF intensity: {vpcf_images.mean():.4f}")
print(f"Scan position range: ({atomic_positions[:, 0].min():.1f}, {atomic_positions[:, 1].min():.1f}) to ({atomic_positions[:, 0].max():.1f}, {atomic_positions[:, 1].max():.1f})")
```

### Export Specific Samples

```python
# Export only high-confidence samples or specific indices
sample_ids_to_export = [0, 5, 10, 15, 20]

for sample_id in sample_ids_to_export:
    vpcf_data = mapper.get_vpcf_by_sample_id(sample_id)
    # Use matplotlib or PIL to save
    # ...
```

## Example Outputs

After running the analysis, you'll get:

1. **vPCF Montages** - PNG images showing grids of vPCF patterns for each cluster
2. **Metadata CSVs** - Tables with sample IDs, cluster assignments, atomic positions
3. **Mapping Reports** - Complete sample-to-cluster mappings with coordinates
4. **Comparison Reports** - Model agreement statistics and disagreement analysis

## Next Steps

1. Run the example script: `python example_visual_mapping.py`
2. Check the generated visualization outputs in `results/visual_analysis/`
3. Use the mapping reports to identify interesting clusters for further analysis
4. Compare DEC and IDEC results to understand model differences
5. Map notable vPCF patterns back to STEM images for experimental validation

## Citation

If you use this mapping module in your research, please cite the original vPCF and DEC/IDEC papers:
- DEC: Xie, J., Girshick, R., & Farhadi, A. (2016). Unsupervised deep embedding for clustering analysis
- IDEC: Guo, X., Gao, X., Yin, X., & Fang, B. (2017). Deep embedded clustering with data augmentation
