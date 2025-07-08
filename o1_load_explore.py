# 01_load_explore.py
import pandas as pd
import logging
from pathlib import Path
import json

def load_and_explore_data(data_path, results_dir):
    """Loads data, performs initial exploration, creates binary attack label,
       and saves basic exploration info."""
    logging.info(f"PHASE 1: Loading data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error(f"Dataset file not found at {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    logging.info(f"Data loaded. Shape: {df.shape}")

    # Basic check for required columns
    required_cols = ['label', 'type']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Missing one or more required columns: {required_cols}. Found: {df.columns.tolist()}")
        raise ValueError(f"Dataset must contain 'label' and 'type' columns.")

    exploration_info = {
        "shape": df.shape,
        "label_distribution_summary": df['label'].value_counts(normalize=True).to_dict(),
        "type_distribution_summary": df['type'].value_counts(normalize=True).to_dict(),
        "data_types": {col: str(df[col].dtype) for col in df.columns},
        "missing_values_summary": df.isnull().sum().to_dict()
    }

    # Create binary attack label
    if df['label'].dtype == 'object':
        df['attack_label'] = df['label'].astype(str).str.lower().apply(lambda x: 0 if x == 'normal' else 1)
    elif pd.api.types.is_numeric_dtype(df['label']): # Assuming 0 is normal, anything else is attack
        df['attack_label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    else:
        logging.warning(f"Interpreting 'label' column (dtype: {df['label'].dtype}) for binary target. Attempting conversion to string and then 'normal' vs 'attack'.")
        try:
            df['attack_label'] = df['label'].astype(str).str.lower().apply(lambda x: 0 if x == 'normal' else 1)
        except Exception as e:
            logging.error(f"Could not create 'attack_label' from 'label' column: {e}")
            raise

    exploration_info["binary_attack_label_distribution_summary"] = df['attack_label'].value_counts(normalize=True).to_dict()
    logging.info(f"Binary 'attack_label' created. Distribution summary logged.")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    # Convert int64 to int for JSON serialization if present in shapes
    exploration_info["shape"] = [int(s) for s in exploration_info["shape"]]
    for key, value in exploration_info["missing_values_summary"].items():
        exploration_info["missing_values_summary"][key] = int(value)


    with open(results_path / "01_data_exploration_summary.json", 'w') as f:
        json.dump(exploration_info, f, indent=4)
    logging.info(f"Data exploration summary saved to {results_path / '01_data_exploration_summary.json'}")
    logging.info("PHASE 1: Load and Explore COMPLETE.")
    return df