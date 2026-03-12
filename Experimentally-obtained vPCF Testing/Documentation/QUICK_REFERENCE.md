# Sample Metadata Lookup - Quick Reference Card

## The Problem You Solved
Map sample index → complete vPCF metadata (atomic positions, origins, shapes, crystal type)

## 3-Minute Quickstart

### Option 1: High-Level Interface (Recommended)
```python
from sample_metadata_lookup import SampleMetadataManager

# Initialize
manager = SampleMetadataManager("data/vPCF_test_2.h5")

# Get complete info for sample 123
info = manager.get_sample_info(123)
print(info['atomic_positions'])  # [1624.0, 1243.0]
print(info['vpcf_origin'])       # [500.0, 498.0]
print(info['vpcf_shape'])        # (999, 999)

# Export batch metadata
manager.export_sample_metadata(
    sample_ids=[0, 15, 42, 100, 123],
    output_file="samples.csv"
)

# Show what's available
manager.print_summary()
```

### Option 2: Direct H5 Access (Advanced)
```python
from vpcf_cluster_visual_mapping import VPCFClusterMapper

mapper = VPCFClusterMapper("data/vPCF_test_2.h5")

# Get metadata
metadata = mapper.get_sample_metadata(123, include_vpcf_image=False)

# Get metadata WITH image array
metadata = mapper.get_sample_metadata(123, include_vpcf_image=True)
vpcf_img = metadata['vpcf_image']  # Shape (999, 999)

# Inspect H5 structure
mapper.print_h5_structure()
```

---

## Available Data Per Sample

For any sample index (0-7204):

| Data | Type | Shape | Example |
|------|------|-------|---------|
| `atomic_positions` | float16 | (2,) | [1623.0, 1344.0] |
| `vpcf_origin` | float16 | (2,) | [500.0, 498.0] |
| `vpcf_image` | float16 | (999, 999) | Diffraction pattern |
| `vpcf_peaks` | float16 | (380,) | Peak list |
| `vpcf_peaks_shape` | int16 | (2,) | [76, 5] |
| `crystal_label` | str | N/A | "Monoclinic_001" (if labels available) |

---

## Common Tasks

### Task 1: Look up sample 123
```python
info = manager.get_sample_info(123)
print(info)  # All metadata for sample 123
```

### Task 2: Export cluster samples to CSV
```python
manager.export_sample_metadata(
    sample_ids=[0, 15, 42, 100],
    output_file="cluster_metadata.csv"
)
# Creates CSV with columns: sample_id, atomic_x, atomic_y, vpcf_origin_x, vpcf_origin_y, vpcf_shape
```

### Task 3: Get vPCF image for visualization
```python
# Method A: Include in get_sample_info
info = manager.get_sample_info(123, include_image=True)
vpcf = info['vpcf_image']

# Method B: Use mapper directly
vpcf_data = mapper.get_vpcf_by_sample_id(123, return_metadata=True)
vpcf = vpcf_data['vpcf_image']
```

### Task 4: Search by crystal type (if labels available)
```python
samples = manager.search_by_crystal_type("Tetragonal")  # Returns [list of indices]

crystal_types = manager.get_crystal_types()  # Returns {type: [indices]}
```

### Task 5: Create summary report
```python
manager.print_summary()
# Shows: total samples, available metadata, crystal types breakdown
```

### Task 6: Filter samples by custom criteria
```python
# Get all samples where atomic_x > 1600
high_x_samples = mapper.get_samples_by_metadata_filter(
    lambda sid: mapper.get_sample_metadata(sid)['atomic_positions'][0] > 1600
)
```

---

## Files Reference

| File | Purpose | When to Use |
|------|---------|------------|
| `sample_metadata_lookup.py` | High-level unified interface | Most use cases |
| `vpcf_cluster_visual_mapping.py` | Extended mapper with metadata methods | Advanced H5 access |
| `h5_metadata_inspector.py` | H5 diagnostic tool | Exploring file structure |
| `test_sample_metadata.py` | Test suite | Verification |
| `sample_usage_examples.py` | Practical examples | Learning |

---

## API Reference

### SampleMetadataManager

```python
manager = SampleMetadataManager(h5_filepath, external_labels_csv=None, verbose=True)

# Core methods
info = manager.get_sample_info(sample_id, include_image=False)
samples = manager.search_by_crystal_type(crystal_type)
types = manager.get_crystal_types()
result = manager.export_sample_metadata(sample_ids, output_file, include_images=False)
report = manager.create_summary_report()
manager.print_summary()
```

### VPCFClusterMapper (New Methods)

```python
mapper = VPCFClusterMapper(h5_filepath, verbose=True)

# New methods
metadata = mapper.get_sample_metadata(sample_id, include_vpcf_image=False)
structure = mapper.inspect_h5_full_structure()
mapper.print_h5_structure()
samples = mapper.get_samples_by_metadata_filter(filter_func)
```

---

## Example: Real-World Integration

```python
from vpcf_cluster_visual_mapping import VPCFClusterMapper
from sample_metadata_lookup import SampleMetadataManager

# Set up
mapper = VPCFClusterMapper("data/vPCF_test_2.h5")
manager = SampleMetadataManager("data/vPCF_test_2.h5")

# Load clustering results
mapper.load_inspection_results("results/dec/vpcf/DEC_detailed_cluster_report.csv", source_name="dec")

# Analyze cluster 0
cluster_samples = mapper.get_cluster_samples(0, source="dec")
print(f"Cluster 0 has {len(cluster_samples)} samples")

# Export metadata for cluster
manager.export_sample_metadata(
    sample_ids=cluster_samples,
    output_file="cluster_0_metadata.csv"
)

# For each sample, get vPCF + metadata
for sample_id in cluster_samples[:5]:  # First 5
    info = manager.get_sample_info(sample_id)
    vpcf = mapper.get_vpcf_by_sample_id(sample_id)['vpcf_image']
    
    print(f"Sample {sample_id}:")
    print(f"  Position: {info['atomic_positions']}")
    print(f"  vPCF shape: {vpcf.shape}")
```

---

## Testing

Run the comprehensive test suite:
```bash
python test_sample_metadata.py --full
```

Test specific components:
```bash
python test_sample_metadata.py --test-manager
python test_sample_metadata.py --test-mapper
python test_sample_metadata.py --test-inspector
```

Test specific sample:
```bash
python test_sample_metadata.py --sample 123
```

---

## Key Insights

### Data Organization
- **7205 total samples** indexed 0-7204
- **Atomic positions**: Scan coordinates in original STEM image  
- **vPCF origin**: Typically [500, 498] (center of 999×999 grid)
- **vPCF peaks**: Stored flattened, shape info in peaks_shapes

### Performance
- Single sample lookup: ~1ms
- Batch export (100 samples): ~100ms
- Full 7205 scan (filter all): ~30-60 seconds (slow but works)

### Crystal labels
- Optional external labels via CSV
- System auto-detects at: `data/labels.csv` or `../results/vpcf_label_mapping.csv`
- Works fine without labels

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No external label file found" | OK! System works without labels. Crystal type just unknown. |
| Filter function is slow | Normal for 7205 samples. Use batch export instead. |
| "Sample X out of range" | Use sample indices 0-7204 only. |
| Import error | Ensure h5py, pandas installed: `pip install h5py pandas` |

---

## Next Steps

1. ✓ Try basic lookup: `manager.get_sample_info(123)`
2. ✓ Run tests: `python test_sample_metadata.py --full`
3. ✓ Explore examples: `python sample_usage_examples.py`
4. ✓ Integrate with clustering workflow
5. ✓ Export metadata for analysis

**You're ready to go!** 🚀
