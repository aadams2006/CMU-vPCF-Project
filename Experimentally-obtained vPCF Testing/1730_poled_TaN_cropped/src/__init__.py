"""
Experimentally-obtained vPCF Testing

This package provides tools for loading, preprocessing, and training 
DEC/IDEC clustering models on experimentally obtained vPCF data.

Modules:
- vpcf_data_loader: Data loading and preprocessing utilities
- train_vpcf_models: Training pipeline for DEC/IDEC models
- run_vpcf_test_2_training: Quick-start script for 1730_poled_TaN_cropped files
"""

from .vpcf_data_loader import (
    load_vpcf_file,
    combine_datasets,
    check_dependencies,
    VPCFDataset,
)

__all__ = [
    "load_vpcf_file",
    "combine_datasets", 
    "check_dependencies",
    "VPCFDataset",
]
