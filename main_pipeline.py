import logging
import json
from pathlib import Path
import pandas as pd

from oo_config import CONFIG
from o1_load_explore import load_and_explore_data
from o2_preprocess import preprocess_data
from o3_feature_selection import select_features
from o4_train_evaluate import train_evaluate_feature_set


results_dir_path_for_log = Path(CONFIG["results_dir"])
results_dir_path_for_log.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=CONFIG.get("log_level", logging.INFO),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(results_dir_path_for_log / "pipeline.log", mode='w') # Use the created path
                    ])

def main_workflow():
    logging.info("--- PIPELINE START ---")

    results_path = Path(CONFIG["results_dir"]) 

    try:
        
        logging.info("--- Starting Step 1: Load and Explore Data ---")
        df_raw = load_and_explore_data(CONFIG["data_path"], CONFIG["results_dir"])
        if df_raw is None:
            logging.critical("Data loading failed. Exiting.")
            return
        logging.info("--- Completed Step 1: Load and Explore Data ---")

        
        logging.info("--- Starting Step 2: Preprocess Data ---")
        X_processed, y_binary_target, y_multiclass_target, le_type_obj = \
            preprocess_data(df_raw, CONFIG["features_to_drop_static"], CONFIG["results_dir"])
        if X_processed is None:
            logging.critical("Preprocessing failed. Exiting.")
            return
        logging.info("--- Completed Step 2: Preprocess Data ---")

        
        logging.info("--- Starting Step 3: Feature Selection ---")
        
        global selected_feature_sets
        selected_feature_sets = select_features(
            X_processed, y_binary_target,
            CONFIG["correlation_threshold_for_removal"],
            CONFIG["rf_feature_importance_top_n"],
            CONFIG["random_state"],
            CONFIG["results_dir"]
        )
        if not selected_feature_sets:
            logging.critical("No feature sets created during feature selection. Exiting.")
            return
        logging.info(f"--- Completed Step 3: Feature Selection. {len(selected_feature_sets)} feature sets created. ---")

        all_pipeline_results = {"binary": {}, "multiclass": {}}

        
        logging.info("--- Starting Step 4: Model Training and Evaluation ---")
        for fs_name, fs_content in selected_feature_sets.items():
            logging.info(f"\n--- Processing Feature Set: {fs_name} ({len(fs_content['features_list'])} features) ---")
            X_current_fs_data = fs_content["X_data"]
            
            if X_current_fs_data.empty or X_current_fs_data.shape[1] == 0:
                logging.warning(f"Feature set '{fs_name}' is empty or has no features. Skipping evaluation.")
                all_pipeline_results["binary"][fs_name] = {"Error": "Empty feature set"}
                all_pipeline_results["multiclass"][fs_name] = {"Error": "Empty feature set"}
                continue

            
            logging.info(f"--- Starting BINARY task for {fs_name} ---")
            binary_fs_results = train_evaluate_feature_set(
                X_current_fs_data, y_binary_target, CONFIG["classifiers"],
                task_type="binary", test_size=CONFIG["test_size"],
                random_state=CONFIG["random_state"], smote_k_min=CONFIG["smote_k_neighbors_multi_min"]
            )
            all_pipeline_results["binary"][fs_name] = binary_fs_results
            with open(results_path / f"binary_results_{fs_name}.json", 'w') as f:
                json.dump(binary_fs_results, f, indent=4, default=str)
            logging.info(f"Binary task results for {fs_name} saved.")

            
            logging.info(f"--- Starting MULTI-CLASS task for {fs_name} ---")
            multiclass_target_names = [str(cls) for cls in le_type_obj.classes_]
            multiclass_fs_results = train_evaluate_feature_set(
                X_current_fs_data, y_multiclass_target, CONFIG["classifiers"],
                task_type="multiclass", test_size=CONFIG["test_size"],
                random_state=CONFIG["random_state"], smote_k_min=CONFIG["smote_k_neighbors_multi_min"],
                target_names=multiclass_target_names
            )
            all_pipeline_results["multiclass"][fs_name] = multiclass_fs_results
            with open(results_path / f"multiclass_results_{fs_name}.json", 'w') as f:
                json.dump(multiclass_fs_results, f, indent=4, default=str)
            logging.info(f"Multi-class task results for {fs_name} saved.")

        logging.info("--- All Feature Sets Processed. Completed Step 4. ---")
        
        generate_summary_report(all_pipeline_results, results_path, selected_feature_sets) 

    except Exception as e:
        logging.critical(f"An unhandled error occurred in the main pipeline: {e}", exc_info=True)
    finally:
        logging.info("--- PIPELINE COMPLETE ---")


def generate_summary_report(all_results, results_dir_path, current_selected_feature_sets): 
    logging.info("Generating final summary report...")
    summary_data = []
    
    for task_type, fs_dict in all_results.items():
        for fs_name, model_dict in fs_dict.items():
            if isinstance(model_dict, dict) and "Error" in model_dict and model_dict["Error"] == "Empty feature set":
                 logging.warning(f"Skipping summary for {fs_name} in {task_type} as it was an empty feature set.")
                 continue

            if not isinstance(model_dict, dict): 
                 logging.warning(f"Unexpected model_dict format for {fs_name} in {task_type}. Expected dict, got {type(model_dict)}. Skipping.")
                 continue

            for model_name, metrics in model_dict.items():
                if isinstance(metrics, dict) and "Error" in metrics:
                    logging.warning(f"Skipping {model_name} on {fs_name} for {task_type} due to error: {metrics['Error']}")
                    continue
                if not isinstance(metrics, dict):
                    logging.warning(f"Unexpected metrics format for {model_name} on {fs_name} for {task_type}. Metrics: {metrics}")
                    continue

                num_features = metrics.get("Feature_Count") 
                if num_features is None and fs_name in current_selected_feature_sets:
                     num_features = len(current_selected_feature_sets[fs_name]["features_list"])
                else:
                    num_features = "N/A"


                row = {
                    "Task Type": task_type,
                    "Feature Set": fs_name,
                    "Num Features": num_features,
                    "Model": model_name,
                    "Accuracy": metrics.get("Accuracy"),
                    "Train Time (s)": metrics.get("Train Time (s)"),
                    "Prediction Time (s)": metrics.get("Prediction Time (s)")
                }
                if task_type == "binary":
                    row["F1-Score (Binary)"] = metrics.get("F1-Score")
                    row["AUC-ROC (Binary)"] = metrics.get("AUC-ROC")
                    row["Specificity (Binary)"] = metrics.get("Specificity")
                else: 
                    row["F1-Score (Macro MC)"] = metrics.get("F1-Score (Macro)")
                    row["AUC-ROC (Macro OvR MC)"] = metrics.get("AUC-ROC (Macro OvR)")
                summary_data.append(row)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(results_dir_path / "pipeline_summary_report.csv", index=False)
        logging.info(f"Summary report saved to {results_dir_path / 'pipeline_summary_report.csv'}")
        
        if 'F1-Score (Binary)' in summary_df.columns:
            
            summary_df['F1-Score (Binary)'] = pd.to_numeric(summary_df['F1-Score (Binary)'], errors='coerce')
            top_3_binary = summary_df[summary_df['Task Type'] == 'binary'].nlargest(3, 'F1-Score (Binary)')
            logging.info(f"\nTop 3 Models for Binary Classification (by F1-Score):\n{top_3_binary[['Feature Set', 'Num Features', 'Model', 'Accuracy', 'F1-Score (Binary)', 'AUC-ROC (Binary)']]}")
        
        if 'F1-Score (Macro MC)' in summary_df.columns:
            summary_df['F1-Score (Macro MC)'] = pd.to_numeric(summary_df['F1-Score (Macro MC)'], errors='coerce')
            top_3_multiclass_f1 = summary_df[summary_df['Task Type'] == 'multiclass'].nlargest(3, 'F1-Score (Macro MC)')
            logging.info(f"\nTop 3 Models for Multi-Class Classification (by F1-Score Macro):\n{top_3_multiclass_f1[['Feature Set', 'Num Features', 'Model', 'Accuracy', 'F1-Score (Macro MC)', 'AUC-ROC (Macro OvR MC)']]}")

            summary_df['Accuracy'] = pd.to_numeric(summary_df['Accuracy'], errors='coerce')
            top_3_multiclass_acc = summary_df[summary_df['Task Type'] == 'multiclass'].nlargest(3, 'Accuracy')
            logging.info(f"\nTop 3 Models for Multi-Class Classification (by Accuracy):\n{top_3_multiclass_acc[['Feature Set','Num Features', 'Model', 'Accuracy', 'F1-Score (Macro MC)', 'AUC-ROC (Macro OvR MC)']]}")
    else:
        logging.info("No summary data to generate report.")


if __name__ == "__main__":
    main_workflow()