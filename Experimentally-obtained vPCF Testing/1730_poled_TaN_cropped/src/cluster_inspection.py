"""
Utilities for inspecting and analyzing cluster assignments from DEC/IDEC models.

This module provides functions to:
- Generate detailed cluster inspection reports
- Map vPCF samples to their assigned clusters
- Compare predicted clusters with ground truth labels
- Create cluster composition summaries
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def create_cluster_report(
    labels: np.ndarray,
    model_name: str = "model",
    sample_names: Optional[np.ndarray] = None,
    ground_truth_labels: Optional[np.ndarray] = None,
    save_dir: str = "results"
) -> pd.DataFrame:
    """
    Create a detailed cluster assignment report.
    
    Parameters
    ----------
    labels : np.ndarray
        Predicted cluster labels of shape (n_samples,).
    model_name : str
        Name of the model (for labeling files).
    sample_names : np.ndarray, optional
        Names/identifiers for each sample. If None, uses indices.
    ground_truth_labels : np.ndarray, optional
        Ground truth labels for comparison.
    save_dir : str
        Directory to save the report.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with detailed cluster assignment information.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_samples = len(labels)
    
    # Create report dataframe
    report_data = {
        'sample_idx': np.arange(n_samples),
        'predicted_cluster': labels,
    }
    
    # Add sample names if provided
    if sample_names is not None:
        assert len(sample_names) == n_samples, "sample_names length mismatch"
        report_data['sample_name'] = sample_names
    
    # Add ground truth if provided
    if ground_truth_labels is not None:
        assert len(ground_truth_labels) == n_samples, "ground_truth_labels length mismatch"
        report_data['ground_truth_cluster'] = ground_truth_labels
        
        # Add match column
        report_data['matches_ground_truth'] = labels == ground_truth_labels
    
    report_df = pd.DataFrame(report_data)
    
    # Save report
    report_path = os.path.join(save_dir, f"{model_name}_detailed_cluster_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"Saved detailed cluster report to: {report_path}")
    
    return report_df


def create_cluster_composition(
    labels: np.ndarray,
    model_name: str = "model",
    sample_names: Optional[np.ndarray] = None,
    save_dir: str = "results"
) -> Dict[int, List[int]]:
    """
    Create a mapping of cluster IDs to sample indices/names.
    
    Parameters
    ----------
    labels : np.ndarray
        Predicted cluster labels of shape (n_samples,).
    model_name : str
        Name of the model (for labeling files).
    sample_names : np.ndarray, optional
        Names/identifiers for each sample. If None, uses indices.
    save_dir : str
        Directory to save the composition report.
        
    Returns
    -------
    dict
        Dictionary mapping cluster IDs to lists of sample indices or names.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_samples = len(labels)
    unique_clusters = np.unique(labels)
    
    # Create composition mapping
    composition = {}
    for cluster_id in unique_clusters:
        sample_indices = np.where(labels == cluster_id)[0]
        
        if sample_names is not None:
            composition[int(cluster_id)] = list(sample_names[sample_indices])
        else:
            composition[int(cluster_id)] = list(map(int, sample_indices))
    
    # Save as detailed CSV for each cluster
    composition_rows = []
    for cluster_id in unique_clusters:
        samples = composition[int(cluster_id)]
        sample_count = len(samples)
        percentage = (sample_count / n_samples) * 100
        
        composition_rows.append({
            'cluster_id': int(cluster_id),
            'sample_count': sample_count,
            'percentage': percentage,
            'samples': ';'.join(map(str, samples))
        })
    
    composition_df = pd.DataFrame(composition_rows)
    composition_path = os.path.join(save_dir, f"{model_name}_cluster_composition.csv")
    composition_df.to_csv(composition_path, index=False)
    print(f"Saved cluster composition to: {composition_path}")
    
    # Also save individual files for each cluster
    for cluster_id in unique_clusters:
        samples = composition[int(cluster_id)]
        cluster_samples_df = pd.DataFrame({
            'sample_index': samples if sample_names is None else range(len(samples)),
            'sample_name': samples if sample_names is not None else samples
        })
        cluster_file = os.path.join(save_dir, f"{model_name}_cluster_{cluster_id}_samples.csv")
        cluster_samples_df.to_csv(cluster_file, index=False)
    
    return composition


def create_ground_truth_comparison(
    predicted_labels: np.ndarray,
    ground_truth_labels: np.ndarray,
    model_name: str = "model",
    save_dir: str = "results"
) -> pd.DataFrame:
    """
    Create a comparison between predicted and ground truth clusters.
    
    Parameters
    ----------
    predicted_labels : np.ndarray
        Predicted cluster labels.
    ground_truth_labels : np.ndarray
        Ground truth labels.
    model_name : str
        Name of the model (for labeling files).
    save_dir : str
        Directory to save the comparison.
        
    Returns
    -------
    pd.DataFrame
        Comparison report with accuracy metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    assert len(predicted_labels) == len(ground_truth_labels), "Label length mismatch"
    
    n_samples = len(predicted_labels)
    matches = predicted_labels == ground_truth_labels
    accuracy = np.sum(matches) / n_samples
    
    # Create confusion-like matrix
    unique_pred = np.unique(predicted_labels)
    unique_true = np.unique(ground_truth_labels)
    
    comparison_rows = []
    
    # For each predicted cluster, show distribution of ground truth
    for pred_cluster in sorted(unique_pred):
        mask = predicted_labels == pred_cluster
        pred_count = np.sum(mask)
        
        for true_cluster in sorted(unique_true):
            overlap = np.sum((predicted_labels == pred_cluster) & (ground_truth_labels == true_cluster))
            overlap_pct = (overlap / pred_count * 100) if pred_count > 0 else 0
            
            comparison_rows.append({
                'predicted_cluster': int(pred_cluster),
                'ground_truth_cluster': int(true_cluster),
                'sample_count': overlap,
                'percentage_of_pred_cluster': overlap_pct
            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(save_dir, f"{model_name}_ground_truth_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved ground truth comparison to: {comparison_path}")
    print(f"  Overall accuracy: {accuracy * 100:.2f}%")
    
    # Save summary stats
    summary_path = os.path.join(save_dir, f"{model_name}_comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Cluster Prediction vs Ground Truth Comparison\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Total samples: {n_samples}\n")
        f.write(f"Correctly predicted: {np.sum(matches)} ({accuracy * 100:.2f}%)\n")
        f.write(f"Incorrectly predicted: {n_samples - np.sum(matches)} ({(1 - accuracy) * 100:.2f}%)\n\n")
        f.write(f"Predicted clusters: {len(unique_pred)}\n")
        f.write(f"Ground truth clusters: {len(unique_true)}\n")
    
    return comparison_df


def create_cluster_inspection_report(
    predicted_labels: np.ndarray,
    model_name: str = "model",
    sample_names: Optional[np.ndarray] = None,
    ground_truth_labels: Optional[np.ndarray] = None,
    save_dir: str = "results"
) -> Dict[str, pd.DataFrame]:
    """
    Create a comprehensive cluster inspection report with multiple analyses.
    
    Parameters
    ----------
    predicted_labels : np.ndarray
        Predicted cluster labels.
    model_name : str
        Name of the model.
    sample_names : np.ndarray, optional
        Names/identifiers for each sample.
    ground_truth_labels : np.ndarray, optional
        Ground truth labels for comparison.
    save_dir : str
        Directory to save all reports.
        
    Returns
    -------
    dict
        Dictionary containing all generated reports:
        - 'detailed_report': Detailed sample-level assignments
        - 'composition': Cluster composition summary
        - 'comparison': Ground truth comparison (if available)
    """
    print(f"\nGenerating comprehensive cluster inspection reports for {model_name}...")
    print(f"Save directory: {save_dir}")
    
    results = {}
    
    # Create detailed report
    detailed_report = create_cluster_report(
        predicted_labels,
        model_name=model_name,
        sample_names=sample_names,
        ground_truth_labels=ground_truth_labels,
        save_dir=save_dir
    )
    results['detailed_report'] = detailed_report
    
    # Create cluster composition
    composition = create_cluster_composition(
        predicted_labels,
        model_name=model_name,
        sample_names=sample_names,
        save_dir=save_dir
    )
    results['composition'] = composition
    
    # Create ground truth comparison if available
    if ground_truth_labels is not None:
        comparison = create_ground_truth_comparison(
            predicted_labels,
            ground_truth_labels,
            model_name=model_name,
            save_dir=save_dir
        )
        results['comparison'] = comparison
    
    # Create summary statistics
    unique_clusters = np.unique(predicted_labels)
    summary_stats = {
        'n_samples': len(predicted_labels),
        'n_clusters': len(unique_clusters),
        'cluster_sizes': dict(zip(
            map(int, unique_clusters),
            map(int, [np.sum(predicted_labels == c) for c in unique_clusters])
        ))
    }
    
    # Save summary to text file
    summary_path = os.path.join(save_dir, f"{model_name}_inspection_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Cluster Inspection Report for {model_name}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Total samples: {summary_stats['n_samples']}\n")
        f.write(f"Number of clusters: {summary_stats['n_clusters']}\n\n")
        f.write(f"Cluster Sizes:\n")
        for cluster_id in sorted(summary_stats['cluster_sizes'].keys()):
            size = summary_stats['cluster_sizes'][cluster_id]
            pct = (size / summary_stats['n_samples']) * 100
            f.write(f"  Cluster {cluster_id}: {size} samples ({pct:.1f}%)\n")
    
    print(f"Saved inspection summary to: {summary_path}")
    print(f"Cluster composition: {summary_stats['cluster_sizes']}\n")
    
    return results


def export_cluster_assignments_with_metadata(
    predicted_labels: np.ndarray,
    model_name: str = "model",
    indices_to_export: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, np.ndarray]] = None,
    save_dir: str = "results"
) -> pd.DataFrame:
    """
    Export cluster assignments along with optional metadata.
    
    Parameters
    ----------
    predicted_labels : np.ndarray
        Predicted cluster labels.
    model_name : str
        Name of the model.
    indices_to_export : np.ndarray, optional
        Specific sample indices to export.
    metadata : dict, optional
        Dictionary mapping column names to arrays of metadata values.
    save_dir : str
        Directory to save the export.
        
    Returns
    -------
    pd.DataFrame
        Exported data with cluster assignments and metadata.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_samples = len(predicted_labels)
    
    # Determine which samples to export
    if indices_to_export is not None:
        export_mask = indices_to_export
        export_labels = predicted_labels[export_mask]
    else:
        export_mask = np.arange(n_samples)
        export_labels = predicted_labels
    
    # Create base dataframe
    export_data = {
        'sample_index': export_mask,
        'predicted_cluster': export_labels
    }
    
    # Add metadata if provided
    if metadata is not None:
        for col_name, col_values in metadata.items():
            if len(col_values) == n_samples:
                export_data[col_name] = col_values[export_mask]
            else:
                print(f"Warning: Skipping metadata column '{col_name}' (length mismatch)")
    
    export_df = pd.DataFrame(export_data)
    
    # Save export
    export_path = os.path.join(save_dir, f"{model_name}_cluster_assignments.csv")
    export_df.to_csv(export_path, index=False)
    print(f"Saved cluster assignments with metadata to: {export_path}")
    
    return export_df


if __name__ == "__main__":
    # Example usage
    print("Cluster inspection utilities loaded successfully")
    print("Use create_cluster_inspection_report() to generate comprehensive reports")
