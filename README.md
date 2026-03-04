# A Comparative Study of Feature Selection Techniques for Efficient Network Intrusion Detection in IoT Systems
This repository contains the modular machine learning pipeline developed for our research on Network Intrusion Detection Systems (NIDS) in IoT environments. The pipeline follows a modular experimental workflow from preprocessing to evaluation.

## Dataset
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
6. `baseline_pipeline.py`:Train all 8 classifiers using the full original feature set (928+ features after preprocessing).
7. `main_pipeline.py`:Main script to execute the full experimental workflow.

## Usage Instructions

```bash
git clone https://github.com/AhmadIAlawneh/intrusion-detection-pipeline.git
cd intrusion-detection-pipeline
pip install -r requirements.txt
```
Place the train_test_network.csv (from the ToN-IoT source) in the root directory.
Execute the Pipeline
```bash
python main_pipeline.py
```

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- xgboost
- matplotlib
- seaborn

## Methodology

1. Data Exploration and Distribution Analysis  
   - Statistical inspection and class distribution analysis of the network dataset.

2. Data Cleaning and Preprocessing  
   - Handling missing values and inconsistencies.  
   - Encoding categorical variables using One-Hot Encoding.  
   - Feature scaling and normalization.

3. Feature Selection  
   - Feature ranking using:
     - Pearson Correlation  
     - Spearman Correlation  
     - Chi-Square Test  
     - Mutual Information  

4. Feature Ranking and Top-K Selection  
   - Selection of the top-ranked 15–20 features based on predefined thresholds.

5. Class Imbalance Handling  
   - Application of SMOTE (Synthetic Minority Over-sampling Technique) to balance class distribution.

6. Model Training  
   - Training and hyperparameter configuration of eight machine learning classifiers.

7. Model Evaluation  
   - Performance assessment using Accuracy, Precision, Recall, and F1-score.

8. Comparative Analysis  
   - Performance comparison between models trained on the full post-preprocessing feature set (900+ features) and the reduced feature subsets.
## Citation

If you use this repository, please cite:

Alawneh, A. (2025).INTRUSION DETECTION IN IOT USING MACHINE LEARNING: A FOCUS 
ON THE NETWORK LAYER WITH THE TON-IOT DATASET . MSc Thesis.

Dataset:
ToN-IoT Dataset – UNSW Canberra Cyber.

## Note
This repository is a research-based implementation of the suggested pipeline.
Some parameters and results may differ from the MSc thesis because of refactoring, reproducibility, or lengthy experiments.
