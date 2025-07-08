
# baseline_pipeline.py
import logging
import json
from pathlib import Path
import pandas as pd

from oo_config import CONFIG
from o1_load_explore import load_and_explore_data
from o2_preprocess import preprocess_data
from o4_train_evaluate import train_evaluate_feature_set

def baseline_workflow():
    logging.info("--- BASELINE PIPELINE START ---")
    results_path = Path(CONFIG["results_dir"])
    results_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load and explore data
        logging.info("--- Step 1: Load and Explore Data ---")
        df_raw = load_and_explore_data(CONFIG["data_path"], CONFIG["results_dir"])
        if df_raw is None:
            logging.critical("Data loading failed. Exiting.")
            return

        # Preprocess data
        logging.info("--- Step 2: Preprocess Data ---")
        X_processed, y_binary_target, y_multiclass_target, le_type_obj =             preprocess_data(df_raw, CONFIG["features_to_drop_static"], CONFIG["results_dir"])
        if X_processed is None:
            logging.critical("Preprocessing failed. Exiting.")
            return

        # Create feature set using all original features
        selected_feature_sets = {
            "OriginalFeaturesOnly": {
                "X_data": X_processed,
                "features_list": list(X_processed.columns)
            }
        }

        all_pipeline_results = {"binary": {}, "multiclass": {}}

        # Train and evaluate using original features only
        for fs_name, fs_content in selected_feature_sets.items():
            X_current_fs_data = fs_content["X_data"]

            logging.info(f"--- Binary Task for Feature Set: {fs_name} ---")
            binary_results = train_evaluate_feature_set(
                X_current_fs_data, y_binary_target, CONFIG["classifiers"],
                task_type="binary", test_size=CONFIG["test_size"],
                random_state=CONFIG["random_state"], smote_k_min=0
            )
            all_pipeline_results["binary"][fs_name] = binary_results
            with open(results_path / f"binary_results_baseline_{fs_name}.json", 'w') as f:
                json.dump(binary_results, f, indent=4, default=str)

            logging.info(f"--- Multiclass Task for Feature Set: {fs_name} ---")
            multiclass_results = train_evaluate_feature_set(
                X_current_fs_data, y_multiclass_target, CONFIG["classifiers"],
                task_type="multiclass", test_size=CONFIG["test_size"],
                random_state=CONFIG["random_state"], smote_k_min=0,
                target_names=[str(cls) for cls in le_type_obj.classes_]
            )
            all_pipeline_results["multiclass"][fs_name] = multiclass_results
            with open(results_path / f"multiclass_results_baseline_{fs_name}.json", 'w') as f:
                json.dump(multiclass_results, f, indent=4, default=str)

        logging.info("--- BASELINE PIPELINE COMPLETE ---")

    except Exception as e:
        logging.critical(f"Unhandled exception in baseline pipeline: {e}", exc_info=True)

# Configure logging
logging.basicConfig(level=CONFIG.get("log_level", logging.INFO),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(Path(CONFIG["results_dir"]) / "baseline_pipeline.log", mode='w')
                    ])

if __name__ == "__main__":
    baseline_workflow()
