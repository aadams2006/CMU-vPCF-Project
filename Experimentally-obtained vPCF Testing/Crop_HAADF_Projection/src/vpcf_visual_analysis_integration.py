"""
Integration guide: How to use vpcf_cluster_visual_mapping with the training pipeline.

This module shows how to:
1. Generate inspection results from DEC/IDEC models
2. Create visual analysis reports automatically
3. Compare DEC vs IDEC cluster assignments visually
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from vpcf_cluster_visual_mapping import VPCFClusterMapper


def generate_inspection_results_for_trained_model(
    h5_filepath: Union[str, Path],
    model_weights_path: Union[str, Path],
    model_type: str = "dec",
    output_dir: Union[str, Path] = "results",
    max_samples: Optional[int] = None
) -> Path:
    """
    Generate inspection results CSV from a trained DEC/IDEC model.
    
    This function:
    1. Loads the trained model
    2. Predicts cluster assignments for all vPCF samples
    3. Creates a detailed inspection report CSV
    
    Parameters
    ----------
    h5_filepath : str or Path
        Path to vPCF_test_2.h5 file
    model_weights_path : str or Path
        Path to saved model weights
    model_type : str
        Type of model: "dec" or "idec"
    output_dir : str or Path
        Directory to save inspection results
    max_samples : int, optional
        Maximum samples to process (for testing)
        
    Returns
    -------
    Path
        Path to generated inspection results CSV
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from vpcf_data_loader import load_h5_images_only, flatten_images, normalize_features
    
    print(f"Generating inspection results for {model_type.upper()} model...")
    
    # Load data
    print("Loading vPCF data...")
    h5_file = Path(h5_filepath)
    vpcf_images = load_h5_images_only(h5_file, max_frames=max_samples, verbose=True)
    
    # Extract features (matching training configuration)
    features = flatten_images(vpcf_images)
    features = normalize_features(features, method="minmax")
    
    print(f"Features shape: {features.shape}")
    
    # Load and evaluate model (this depends on your specific model implementation)
    # For now, create a template - you'll need to adapt this based on your model structure
    try:
        if model_type.lower() == "dec":
            from src.DEC import DEC
            # Load model from weights
            n_clusters = 10  # This should match your training configuration
            model = DEC(
                dims=[features.shape[1], 500, 500, 2000, n_clusters],
                n_clusters=n_clusters
            )
            # Load weights from checkpoint
            model.load_weights(str(model_weights_path))
            
            # Get cluster predictions
            from tensorflow.keras.models import Model
            encoder = Model(
                inputs=model.encoder.input,
                outputs=model.encoder.get_layer(index=-2).output
            )
            encoded = encoder.predict(features)
            labels = model.clustering_layer.predict(encoded).argmax(axis=1)
            
        elif model_type.lower() == "idec":
            from src.IDEC import IDEC
            n_clusters = 10  # This should match your training configuration
            model = IDEC(
                dims=[features.shape[1], 500, 500, 2000, n_clusters],
                n_clusters=n_clusters
            )
            model.load_weights(str(model_weights_path))
            
            # Get cluster predictions
            from tensorflow.keras.models import Model
            encoder = Model(
                inputs=model.encoder.input,
                outputs=model.encoder.get_layer(index=-2).output
            )
            encoded = encoder.predict(features)
            labels = model.clustering_layer.predict(encoded).argmax(axis=1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create inspection report
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_df = pd.DataFrame({
            'sample_idx': np.arange(len(labels)),
            'predicted_cluster': labels
        })
        
        output_file = output_dir / f"{model_type.upper()}_detailed_cluster_report.csv"
        report_df.to_csv(output_file, index=False)
        
        print(f"Saved inspection results to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error generating inspection results: {e}")
        print("Note: This function requires a trained model. Run training first.")
        return None


def compare_dec_vs_idec_clusters(
    h5_filepath: Union[str, Path],
    dec_results_path: Union[str, Path],
    idec_results_path: Union[str, Path],
    output_dir: Union[str, Path] = "results/comparison"
) -> pd.DataFrame:
    """
    Compare cluster assignments between DEC and IDEC models with visual output.
    
    Creates:
    - Comparison CSV showing sample assignments in both models
    - Visual report of differences
    - Cluster agreement statistics
    
    Parameters
    ----------
    h5_filepath : str or Path
        Path to vPCF_test_2.h5
    dec_results_path : str or Path
        Path to DEC inspection results
    idec_results_path : str or Path
        Path to IDEC inspection results
    output_dir : str or Path
        Output directory for comparison
        
    Returns
    -------
    pd.DataFrame
        Comparison report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load both results
    dec_df = pd.read_csv(dec_results_path)
    idec_df = pd.read_csv(idec_results_path)
    
    # Create comparison
    comparison = pd.DataFrame({
        'sample_id': dec_df['sample_idx'],
        'dec_cluster': dec_df['predicted_cluster'],
        'idec_cluster': idec_df['predicted_cluster']
    })
    
    comparison['clusters_match'] = comparison['dec_cluster'] == comparison['idec_cluster']
    
    # Calculate statistics
    match_rate = comparison['clusters_match'].mean() * 100
    n_samples = len(comparison)
    n_different = (~comparison['clusters_match']).sum()
    
    print(f"\nDEC vs IDEC Cluster Comparison:")
    print(f"  Total samples: {n_samples}")
    print(f"  Matching assignments: {n_samples - n_different} ({match_rate:.2f}%)")
    print(f"  Different assignments: {n_different}")
    
    # Save comparison
    comp_file = output_dir / "DEC_vs_IDEC_comparison.csv"
    comparison.to_csv(comp_file, index=False)
    print(f"  Saved comparison to: {comp_file}")
    
    # Initialize mapper to add vPCF metadata
    mapper = VPCFClusterMapper(h5_filepath, verbose=False)
    mapper.load_inspection_results(dec_results_path, source_name="dec")
    mapper.load_inspection_results(idec_results_path, source_name="idec")
    
    # Create visual comparison for mismatched clusters
    print(f"\nGenerating visual analysis for differing assignments...")
    
    differing_samples = comparison[~comparison['clusters_match']]['sample_id'].values
    
    # Export samples where DEC and IDEC disagree
    if len(differing_samples) > 0:
        # Group specific samples with disagreement
        analysis_dir = output_dir / "samples_with_disagreement"
        analysis_dir.mkdir(exist_ok=True)
        
        for sample_id in differing_samples[:10]:  # Limit to first 10 for analysis
            dec_cluster = comparison[comparison['sample_id'] == sample_id]['dec_cluster'].values[0]
            idec_cluster = comparison[comparison['sample_id'] == sample_id]['idec_cluster'].values[0]
            
            vpcf_data = mapper.get_vpcf_by_sample_id(sample_id, return_metadata=True)
            
            # Save sample info
            sample_info = {
                'sample_id': sample_id,
                'dec_cluster': dec_cluster,
                'idec_cluster': idec_cluster,
                'disagreement': f"DEC:{dec_cluster} vs IDEC:{idec_cluster}"
            }
            
            if 'atomic_positions' in vpcf_data:
                sample_info['atomic_x'] = vpcf_data['atomic_positions'][0]
                sample_info['atomic_y'] = vpcf_data['atomic_positions'][1]
            
            sample_df = pd.DataFrame([sample_info])
            sample_df.to_csv(
                analysis_dir / f"sample_{sample_id}_disagreement.csv",
                index=False
            )
    
    return comparison


def create_integrated_visual_analysis(
    h5_filepath: Union[str, Path],
    dec_results_path: Optional[Union[str, Path]] = None,
    idec_results_path: Optional[Union[str, Path]] = None,
    output_base_dir: Union[str, Path] = "results/integrated_analysis"
) -> None:
    """
    Create a complete integrated visual analysis combining DEC and IDEC results.
    
    This is a one-stop function that:
    1. Creates individual cluster analyses for both models
    2. Compares the models side-by-side
    3. Generates a comprehensive report
    
    Parameters
    ----------
    h5_filepath : str or Path
        Path to vPCF_test_2.h5
    dec_results_path : str or Path, optional
        Path to DEC results CSV
    idec_results_path : str or Path, optional
        Path to IDEC results CSV
    output_base_dir : str or Path
        Base directory for all analysis outputs
    """
    from vpcf_cluster_visual_mapping import create_cluster_visual_analysis
    
    output_base = Path(output_base_dir)
    
    print("\n" + "="*70)
    print("INTEGRATED VISUAL ANALYSIS")
    print("="*70)
    
    # DEC analysis
    if dec_results_path is not None:
        dec_path = Path(dec_results_path)
        if dec_path.exists():
            print("\n1. Analyzing DEC clusters...")
            create_cluster_visual_analysis(
                h5_filepath=h5_filepath,
                inspection_csv=dec_results_path,
                output_dir=output_base / "dec_clusters",
                source_name="dec",
                max_samples_per_cluster=5
            )
        else:
            print(f"Warning: DEC results not found at {dec_path}")
    
    # IDEC analysis
    if idec_results_path is not None:
        idec_path = Path(idec_results_path)
        if idec_path.exists():
            print("\n2. Analyzing IDEC clusters...")
            create_cluster_visual_analysis(
                h5_filepath=h5_filepath,
                inspection_csv=idec_results_path,
                output_dir=output_base / "idec_clusters",
                source_name="idec",
                max_samples_per_cluster=5
            )
        else:
            print(f"Warning: IDEC results not found at {idec_path}")
    
    # Comparative analysis
    if dec_results_path is not None and idec_results_path is not None:
        dec_path = Path(dec_results_path)
        idec_path = Path(idec_results_path)
        if dec_path.exists() and idec_path.exists():
            print("\n3. Comparing DEC vs IDEC...")
            compare_dec_vs_idec_clusters(
                h5_filepath=h5_filepath,
                dec_results_path=dec_results_path,
                idec_results_path=idec_results_path,
                output_dir=output_base / "comparison"
            )
    
    print("\n" + "="*70)
    print(f"Analysis complete! Check {output_base} for results.")
    print("="*70)


def main():
    """Example of integrated workflow."""
    h5_file = Path("data/Crop_HAADF_Projection_80pixels.h5")
    
    # Check for inspection results
    dec_results = Path("results/dec/vpcf/DEC_detailed_cluster_report.csv")
    idec_results = Path("results/idec/vpcf/IDEC_detailed_cluster_report.csv")
    
    if h5_file.exists():
        create_integrated_visual_analysis(
            h5_filepath=h5_file,
            dec_results_path=dec_results if dec_results.exists() else None,
            idec_results_path=idec_results if idec_results.exists() else None,
            output_base_dir="results/integrated_analysis"
        )
    else:
        print(f"Error: H5 file not found at {h5_file}")


if __name__ == "__main__":
    main()
