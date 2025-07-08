# 02_preprocess.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def preprocess_data(df, static_drops, results_dir):
    logging.info("PHASE 2: Starting data preprocessing...")

    df_processed = df.drop(columns=static_drops, errors='ignore')
    logging.info(f"Dropped static columns: {static_drops}. Remaining columns: {len(df_processed.columns)}")

    # Impute missing values *before* type checks or conversions
    if df_processed.isnull().sum().sum() > 0:
        # Impute numerical with 0, categorical with 'missing' or mode
        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col].fillna(0, inplace=True)
                else: # object/categorical
                    df_processed[col].fillna('missing', inplace=True) # Using 'missing' category
        logging.info("Filled missing values (numerical with 0, categorical with 'missing').")
    else:
        logging.info("No missing values found to fill.")

    # Ensure target columns are present before feature extraction
    if 'attack_label' not in df_processed.columns:
        logging.error("'attack_label' not found in DataFrame after initial drops.")
        raise KeyError("'attack_label' is missing.")
    if 'type' not in df_processed.columns:
        logging.error("'type' not found in DataFrame after initial drops.")
        raise KeyError("'type' is missing.")

    feature_cols = [col for col in df_processed.columns if col not in ['label', 'type', 'attack_label']]
    X_features = df_processed[feature_cols].copy()
    logging.info(f"Initial number of features for processing: {len(X_features.columns)}")


    # Boolean-like conversion (as in thesis)
    bool_like_cols_from_thesis = ['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_resumed', 'ssl_established']
    # Add other potential boolean-like columns if they appear as objects
    # For robustness, identify potential bool-like cols by unique values if they are object type
    potential_bool_cols = []
    for col in X_features.select_dtypes(include='object').columns:
        unique_vals = X_features[col].astype(str).str.lower().unique()
        if all(val in ['true', 'false', 't', 'f', 'yes', 'no', '1', '0', '-', 'missing'] for val in unique_vals): # added 'missing'
            if col not in bool_like_cols_from_thesis: # Prioritize thesis list
                potential_bool_cols.append(col)

    bool_conversion_map = {'true': 1, 'false': 0, 't': 1, 'f': 0, 'yes': 1, 'no': 0, '1':1, '0':0, '-': 0, 'missing':0} # map missing to 0
    
    converted_bool_cols_count = 0
    for col in list(set(bool_like_cols_from_thesis + potential_bool_cols)): # Use set to avoid duplicates
        if col in X_features.columns and X_features[col].dtype == 'object':
            X_features[col] = X_features[col].astype(str).str.lower().map(bool_conversion_map).fillna(0).astype(int)
            converted_bool_cols_count += 1
    if converted_bool_cols_count > 0:
        logging.info(f"Converted {converted_bool_cols_count} boolean-like object columns to int.")


    # One-Hot Encoding for remaining categorical features
    categorical_features_for_ohe = X_features.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features_for_ohe:
        logging.info(f"One-Hot Encoding {len(categorical_features_for_ohe)} categorical features: {categorical_features_for_ohe}")
        for col in categorical_features_for_ohe:
            if X_features[col].nunique() > 100: # Thesis specified warning at >100
                logging.warning(f"Categorical feature '{col}' has high cardinality ({X_features[col].nunique()}). This can lead to a very large feature space.")

        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None) # drop=None to align with typical usage unless explicitly first category
        encoded_cols_array = one_hot_encoder.fit_transform(X_features[categorical_features_for_ohe])
        encoded_df = pd.DataFrame(encoded_cols_array, columns=one_hot_encoder.get_feature_names_out(categorical_features_for_ohe), index=X_features.index)
        
        X_features.drop(columns=categorical_features_for_ohe, inplace=True)
        X_features = pd.concat([X_features, encoded_df], axis=1)
        logging.info(f"One-Hot Encoding complete. New total feature count: {len(X_features.columns)}")
    else:
        logging.info("No remaining categorical features for one-hot encoding.")

    # Label Encoding for multi-class target 'type'
    le_type = LabelEncoder()
    # Ensure 'type' column is string for consistent label encoding
    y_multiclass_encoded = le_type.fit_transform(df_processed['type'].astype(str))
    type_mapping = {cls_name: int(cls_code) for cls_name, cls_code in zip(le_type.classes_, le_type.transform(le_type.classes_))}
    logging.info(f"Multi-class target 'type' encoded. Found {len(le_type.classes_)} classes.")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / "type_mapping.json", 'w') as f:
        json.dump(type_mapping, f, indent=4)
    logging.info(f"Type mapping saved to {results_path / 'type_mapping.json'}")

    # Final check for non-numeric columns in X_features
    non_numeric_cols = X_features.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        logging.error(f"Found non-numeric columns in X_features after preprocessing: {non_numeric_cols}. These must be handled.")
        for col in non_numeric_cols:
            logging.error(f"Column '{col}' dtype: {X_features[col].dtype}, unique values: {X_features[col].unique()[:5]}")
        raise ValueError("Non-numeric columns remain in X_features.")
    
    # Ensure y_binary_target is Series
    y_binary_target = pd.Series(df_processed['attack_label'], name='attack_label')
    y_multiclass_target_series = pd.Series(y_multiclass_encoded, name='type_encoded', index=df_processed.index)


    logging.info(f"Preprocessing COMPLETE. Final feature shape: {X_features.shape}")
    return X_features, y_binary_target, y_multiclass_target_series, le_type