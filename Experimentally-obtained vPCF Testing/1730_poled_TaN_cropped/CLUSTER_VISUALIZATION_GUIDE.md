# Cluster Visualization Guide

## Overview

I've generated **7 high-quality visualizations** showing how your vPCF clusters are distributed across the original image space. All visualizations map the atomic positions (where each vPCF was extracted in the original image) and color them by cluster assignment.

## Visualizations Generated

All images are saved in: `results/`

### DEC Clustering Visualizations

#### 1. **DEC_cluster_scatter.png**
**Type:** Scatter plot
- Each point represents one vPCF sample
- Points are colored by cluster (Red, Blue, Green)
- Shows the exact spatial distribution of each cluster

**What it shows:**
- Cluster 0 (Red): Large dispersed cluster covering most of the image
- Cluster 1 (Blue): Localized in upper-left region (~730, 590)
- Cluster 2 (Green): Localized in lower-middle region (~850, 890)

#### 2. **DEC_cluster_heatmap.png**
**Type:** Individual density heatmaps (3 subplots)
- Separate heatmap for each cluster showing sample density
- Darker colors = more samples in that region
- Good for seeing cluster concentration areas

**What it shows:**
- Where each cluster has highest density
- Cluster 1 & 2 show clear hotspots
- Cluster 0 shows diffuse distribution

#### 3. **DEC_cluster_combined_heatmap.png**
**Type:** Single color-coded heatmap
- All three clusters in one image
- Color blend shows which cluster dominates each region
- Red (Cluster 0), Blue (Cluster 1), Green (Cluster 2)

**What it shows:**
- Visual representation of the original image overlaid with cluster assignments
- Easy to see cluster boundaries and interactions
- White/gray areas = overlapping or mixed clusters

### IDEC Clustering Visualizations

Same three types as DEC but for IDEC clustering:
- **IDEC_cluster_scatter.png** - Scatter plot
- **IDEC_cluster_heatmap.png** - Density heatmaps
- **IDEC_cluster_combined_heatmap.png** - Color-coded heatmap

### Comparison

#### **DEC_vs_IDEC_comparison.png**
**Type:** Side-by-side comparison
- Left: DEC clustering results
- Right: IDEC clustering results
- Same coordinate space for direct comparison

**What it shows:**
- How DEC and IDEC differ in cluster assignments
- Whether one method finds more localized clusters
- Overall agreement/disagreement between methods

## Key Insights from Your Data

### Cluster Distribution (DEC)

| Cluster | Count | Region | Type |
|---------|-------|--------|------|
| 0 | 5,532 | X: [3-1671], Y: [11-1674] | Dispersed (background) |
| 1 | 891 | X: [419-1085], Y: [314-838] | Localized heatspot |
| 2 | 782 | X: [174-1249], Y: [537-1318] | Intermediate pattern |

### Spatial Patterns

- **Cluster 1**: Forms a tight cluster in one region - likely a distinct feature/phase
- **Cluster 2**: Distributed but concentrated - possibly related structure
- **Cluster 0**: Fills remaining space - could be background or matrix phase

## How to Interpret the Visualizations

### Reading the Scatter Plot
- Each dot = one vPCF sample
- Position = where it was extracted in the original image
- Color = which cluster it belongs to
- Dense regions show where that cluster is concentrated

### Reading the Heatmap
- Bright areas = high sample density
- Dark areas = low sample density
- Helps identify cluster hotspots and boundaries

### Reading the Combined Heatmap
- Pure colors (red/blue/green) = single cluster dominance
- Mixed colors (purple/cyan/yellow) = overlapping clusters
- Smooth gradients = gradual cluster transitions
- Sharp boundaries = distinct cluster separation

## Use Cases

### 1. Material Phase Identification
If clusters represent different crystal phases:
- Cluster 1 & 2 locations show where distinct phases appear
- Cluster 0 might be the bulk/matrix phase

### 2. Defect Localization
If clusters represent different defect types:
- Visualization shows defect hotspots
- Can identify regions requiring further analysis

### 3. Comparing clustering methods
Use the DEC vs IDEC comparison to:
- See if IDEC fine-tunes cluster boundaries
- Identify regions where methods agree/disagree
- Choose the most reliable clustering for your application

## Interactive Usage

### Generate Custom Visualizations

```python
from visualize_clusters import ClusterVisualizer
from pathlib import Path

workspace_root = Path("c:/path/to/workspace")
metadata_file = str(workspace_root / "results" / "all_samples_metadata.json")
clustering_dir = str(workspace_root / "results" / "h5_only" / "dec")

viz = ClusterVisualizer(metadata_file, clustering_dir, method='dec')

# Custom scatter plot
fig = viz.plot_cluster_scatter(
    figsize=(14, 12),
    point_size=30,
    alpha=0.7,
    title='My Custom Title'
)
fig.savefig('custom_scatter.png', dpi=300)

# Custom heatmap
fig = viz.plot_cluster_heatmap(
    resolution=(400, 400),  # Higher resolution
    title='High-Res Heatmap'
)
fig.savefig('custom_heatmap.png', dpi=300)

# Combined visualization
fig = viz.plot_combined_heatmap(
    resolution=(500, 500)
)
fig.savefig('custom_combined.png', dpi=300)

plt.show()
```

### Extract Statistics by Region

```python
# Get all samples in a specific region
region_df = viz.mapper.get_samples_in_region(
    x_min=400, x_max=1100,  # Cluster 1 region
    y_min=314, y_max=838
)

print(f"Found {len(region_df)} samples in region")
print(region_df['cluster'].value_counts())
# Output shows cluster composition in that region
```

### Export Data for Other Tools

```python
# Export coordinates for 3D visualization or other analysis
full_df = viz.mapper.export_cluster_coordinates()
full_df.to_csv('cluster_coordinates_for_other_tools.csv', index=False)

# Now you can import this CSV into:
# - ParaView
# - MATLAB
# - Python Plotly for interactive visualization
# - ArcGIS for geospatial analysis
# - Any 3D visualization tool
```

## File Specifications

- **Format**: PNG images at 150 dpi
- **Resolution**: Sufficient for printing and presentations  
- **Dimensions**: 
  - Scatter plots: ~2400x2000 pixels
  - Heatmaps: Variable based on clustering
  - Comparisons: ~3200x1600 pixels

## Next Steps

1. **View the visualizations** in your results folder
2. **Compare DEC vs IDEC** - which fits your material better?
3. **Analyze specific regions** - use spatial queries to drill down
4. **Extract phase/defect information** - tie clusters back to physical properties
5. **Generate custom visualizations** - adjust resolution, colors, regions as needed

## Troubleshooting

### Colors look washed out
```python
# Increase contrast by adjusting resolution
viz.plot_cluster_heatmap(resolution=(500, 500))  # Higher resolution
```

### Want different color scheme
```python
# Modify in visualize_clusters.py:
# Change: self.colors = plt.cm.tab10(...)
# To:     self.colors = plt.cm.Set3(...)  # or other colormap
```

### Need higher quality images
```python
fig.savefig('output.png', dpi=300, bbox_inches='tight')  # 300 dpi instead of 150
```

## Summary

You now have complete visual representations of:
✅ Where each cluster appears in the original image  
✅ Spatial distribution patterns  
✅ Cluster hotspot locations  
✅ DEC vs IDEC method comparison  
✅ Ready-to-publish quality visualizations  

All derived from your 7,205 vPCF samples with complete coordinate tracking!
