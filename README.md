# Intrusion Detection Pipeline

This repository contains a modular machine learning pipeline for detecting and classifying network-based attacks in IoT environments.

The pipeline follows a basic experimental workflow, including:
- Data loading and exploratory analysis
- Data preprocessing and normalization
- Feature selection to reduce dimensionality
- Training and evaluation of multiple machine learning models (e.g., Logistic Regression, Random Forest, XGBoost)

## Dataset
The dataset consists of network traffic features (e.g., src_ip, dst_port, protocol, flow statistics) commonly used for IoT intrusion detection.  
The dataset can be downloaded from:  
https://shorturl.at/8WWhl

## Installation
```bash
git clone https://github.com/AhmadIAlawneh/intrusion-detection-pipeline.git
cd intrusion-detection-pipeline
pip install -r requirements.txt
```

Note
This repository represents a research-oriented implementation of the proposed pipeline.
Some experimental parameters and results may differ slightly from the MSc thesis due to refactoring, reproducibility considerations, or extended experiments.
