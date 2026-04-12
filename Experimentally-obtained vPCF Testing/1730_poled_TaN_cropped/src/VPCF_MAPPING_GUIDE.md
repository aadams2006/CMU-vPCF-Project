# vPCF to Image Mapping Guide

## Overview

Yes, **direct mapping is absolutely possible!** Your metadata files already contain all the information needed to map vPCF clustering results back to original image coordinates.

### The Key Data You Have

- **`all_samples_metadata.json`**: Contains `atomic_positions` (the [x, y] coordinates in the original image) for each sample
- **`all_samples_vpcf_data.csv`**: vPCF feature data indexed by sample_id
- **`dec_cluster_labels.csv` and `idec_cluster_labels.csv`**: Cluster assignments for each sample
- **Sample IDs**: Unified identifier linking all this data together

### Mapping Strategy

The mapping chain is:
```
vPCF Sample ID → Cluster Assignment → Metadata → Atomic Position in Original Image
```

## Quick Start

### 1. Basic Lookup - Get Cluster and Coordinates for a vPCF

```python
from vpcf_to_image_mapper import VPCFToImageMapper
from pathlib import Path

workspace_root = Path("c:/Users/alexg/Downloads/CMU vPCF Project/Experimentally-obtained vPCF Testing")

mapper = VPCFToImageMapper(
    metadata_file=str(workspace_root / "results" / "all_samples_metadata.json"),
    clustering_results_dir=str(workspace_root / "results" / "h5_only" / "dec"),
    method='dec'
)

# Get info for vPCF sample 100
info = mapper.get_sample_info(100)

print(f"Sample 100:")
print(f"  Cluster: {info['cluster']}")
print(f"  Image coordinates: ({info['atomic_x']}, {info['atomic_y']})")
print(f"  vPCF window: origin ({info['vpcf_origin_x']}, {info['vpcf_origin_y']}) size {info['vpcf_shape']}")
```

Output:
```
Sample 100:
  Cluster: 0
  Image coordinates: (1606.0, 1182.0)
  vPCF window: origin (500.0, 498.0) size [999, 999]
```

### 2. Find All Samples in a Cluster

```python
# Get all samples from cluster 1 with their coordinates
cluster_1_df = mapper.get_cluster_samples(1)

print(f"Cluster 1 has {len(cluster_1_df)} samples")
print(f"Spatial extent: X[{cluster_1_df['atomic_x'].min():.0f}, {cluster_1_df['atomic_x'].max():.0f}]")
print(f"                Y[{cluster_1_df['atomic_y'].min():.0f}, {cluster_1_df['atomic_y'].max():.0f}]")
```

### 3. Spatial Query - Find vPCFs in Image Region

```python
# Find all vPCFs with atomic positions in region
region_df = mapper.get_samples_in_region(
    x_min=500, x_max=800,
    y_min=600, y_max=900
)

print(f"Found {len(region_df)} vPCFs in region")
print(region_df[['sample_id', 'cluster', 'atomic_x', 'atomic_y']])
```

### 4. Export All Data for Visualization

```python
# Export complete mapping as CSV
output_file = workspace_root / "results" / "vpcf_cluster_mapping.csv"
full_df = mapper.export_cluster_coordinates(str(output_file))

# Now you can import this into visualization software
# Columns: sample_id, cluster, atomic_x, atomic_y, vpcf_origin_x, vpcf_origin_y, vpcf_shape, crystal_label
```

### 5. Analyze Cluster Spatial Distribution

```python
stats = mapper.get_cluster_statistics()

for cluster_name, stats in stats.items():
    print(f"{cluster_name}:")
    print(f"  {stats['n_samples']} samples")
    print(f"  Center: ({stats['atomic_x_mean']:.0f}, {stats['atomic_y_mean']:.0f})")
    print(f"  X range: [{stats['atomic_x_min']:.0f}, {stats['atomic_x_max']:.0f}]")
    print(f"  Y range: [{stats['atomic_y_min']:.0f}, {stats['atomic_y_max']:.0f}]")
```

Output shows cluster 0 spans the entire image, while clusters 1 and 2 are localized regions.

## Data Structure Details

### Metadata Format
```json
{
  "total_samples": 7205,
  "samples": {
    "0": {
      "sample_id": 0,
      "atomic_positions": [1623.0, 1344.0],    // ← Your anchor point in original image!
      "vpcf_origin": [500.0, 498.0],          // vPCF window top-left
      "vpcf_shape": [999, 999],               // vPCF dimensions
      "peaks_shape": [76, 5],
      "crystal_label": "N/A"
    },
    ...
  }
}
```

### Cluster Labels Format
```csv
sample_idx,cluster
0,0
1,0
2,0
3,0
...
```

## Your Current Clustering Results

**Loaded from**: `h5_only/dec/` (DEC clustering)

- **Total samples**: 7,205
- **Number of clusters**: 3
- **Cluster 0**: 5,532 samples (widely distributed)
- **Cluster 1**: 891 samples (centered ~733, 588)
- **Cluster 2**: 782 samples (distributed 174-1249, 537-1318)

## Use Cases

### 1. Verify Clustering on Original Image
```python
# Get all cluster 1 samples and visualize their positions on the image
cluster_1_samples = mapper.get_cluster_samples(1)

# Plot on your image using matplotlib
import matplotlib.pyplot as plt
plt.scatter(cluster_1_samples['atomic_x'], 
            cluster_1_samples['atomic_y'], 
            label='Cluster 1')
```

### 2. Investigate Spatially-Localized Features
```python
# Find vPCFs from a specific region + their cluster info
region_vpcfs = mapper.get_samples_in_region(1000, 1500, 1000, 1500)
print(region_vpcfs[['sample_id', 'cluster']].groupby('cluster').size())
```

### 3. Sample Verification
```python
def verify_vpcf_sample(sample_id):
    info = mapper.get_sample_info(sample_id)
    print(f"Sample {sample_id}: Cluster {info['cluster']}")
    print(f"  Look at image position: ({info['atomic_x']}, {info['atomic_y']})")
    print(f"  vPCF window extracted from: ({info['vpcf_origin_x']}, {info['vpcf_origin_y']})")
    return info

verify_vpcf_sample(1000)
```

### 4. Batch Processing
```python
# Process multiple samples
sample_ids = [50, 100, 200, 500, 1000, 5000]
results = []

for sid in sample_ids:
    info = mapper.get_sample_info(sid)
    results.append(info)

# Analyze results
import pandas as pd
df = pd.DataFrame(results)
print(df.groupby('cluster')['sample_id'].count())
```

## Files Created

1. **`vpcf_to_image_mapper.py`** - Main mapping class
   - `get_sample_info(sample_id)` - Get cluster & coordinates
   - `get_cluster_samples(cluster_id)` - Get all samples in cluster
   - `get_samples_in_region(x_min, x_max, y_min, y_max)` - Spatial query
   - `export_cluster_coordinates()` - Export full mapping
   - `get_cluster_statistics()` - Analyze spatial distribution

2. **`vpcf_mapping_examples.py`** - 7 worked examples
   - Query single sample
   - Find all in cluster
   - Spatial queries
   - Export for visualization
   - Integrate with vPCF features
   - Spatial statistics
   - Batch lookup template

## Key Insights

✅ **What's Possible:**
- Direct mapping from vPCF sample → image coordinates
- Cluster-based spatial analysis
- Region-based queries
- Integration with visualization tools
- Batch processing

✅ **Your Data:**
- 7,205 well-indexed samples
- Complete metadata with atomic positions
- 3 clustering results (DEC & IDEC available)
- Ready for visualization/analysis

⚠️ **Note:** All vPCF windows have origin (500, 498) and size (999, 999), suggesting they're all extracted with identical parameters.

## Next Steps

1. Run `vpcf_to_image_mapper.py` to verify setup
2. Run `vpcf_mapping_examples.py` to see all examples
3. Use the mapper in your analysis pipeline
4. Export coordinates for visualization in other tools
