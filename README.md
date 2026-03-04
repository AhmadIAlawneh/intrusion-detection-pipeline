A Comparative Study of Feature Selection Techniques for Efficient Network Intrusion Detection in IoT Systems

This repository contains the modular machine learning pipeline developed for our research on Network Intrusion Detection Systems (NIDS) in IoT environments. The project focuses on balancing high detection accuracy with computational efficiency by reducing high-dimensional feature spaces (from 900+ features to 15-20) using various feature selection techniques.
The pipeline follows a basic experimental workflow, including:
- Data loading and exploratory analysis
- Data preprocessing and normalization
- Feature selection to reduce dimensionality
- Training and evaluation of multiple machine learning models (e.g., Logistic Regression, Random Forest, XGBoost)

## Dataset
## Dataset Information
The experiments are conducted using the **ToN-IoT Dataset**.
- **Original Source:** [UNSW Canberra Cyber - ToN-IoT Datasets](https://research.unsw.edu.au/projects/toniot-datasets)
- **Data Used:** Network Dataset.
## Code Information & Methodology
The pipeline is structured into sequential modules to ensure reproducibility:
1. `01_load_explore.py`: Data ingestion and initial distribution analysis.
2. `02_preprocess.py`: Data cleaning, handling missing values, and One-Hot Encoding for categorical attributes.
3. `03_feature_selection.py`: Implementation of 4 ranking methods (Pearson, Spearman, Chi-Square, and Mutual Information).
4. `04_train_evaluate.py`: Training engine using **SMOTE** for balancing and evaluating 8 classifiers (Random Forest, XGBoost, etc.).
5. `oo_config.py`: Centralized configuration for hyperparameters and thresholds.
6. `baseline_pipeline.py`: Main script to execute the full experimental workflow.

## Installation
```bash
git clone https://github.com/AhmadIAlawneh/intrusion-detection-pipeline.git
cd intrusion-detection-pipeline
pip install -r requirements.txt
```

## Note
This repository is a research-based implementation of the suggested pipeline.
Some parameters and results may differ from the MSc thesis because of refactoring, reproducibility, or lengthy experiments.
