# Deep Embedded Clustering for VPCF Analysis
framework for using deep embedded clustering to cluster vector pair correlation function (VPCFs) from STEM data

## Project Structure
- `data/`: VPCF image data
- `notebooks/`: interactive notebooks for experimentation. `VPCF_Clustering.ipynb` demonstrates the local workflow, while `Kaggle_Training.ipynb` is pre-configured for running on Kaggle's managed hardware.
- `src/`: source code for the IDEC, DEC, and DBSCAN clustering pipelines
- `requirements.txt`: required Python libraries

## Remote training on Kaggle

For detailed, step-by-step instructions on executing the training pipeline in a
Kaggle Notebook (including DEC, IDEC, and the DBSCAN baseline), see
`Kaggle_training_steps.txt`. The accompanying `notebooks/Kaggle_Training.ipynb`
automates the dependency setup, dataset loading, and model execution within the
Kaggle environment.
