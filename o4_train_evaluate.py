# 04_train_evaluate.py
import pandas as pd
import numpy as np
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier # Already imported
from sklearn.base import clone

def train_evaluate_feature_set(
    X_data_fs, y_target, classifiers_dict_config, task_type,
    test_size, random_state, smote_k_min,
    target_names=None # For multiclass classification report
):
    logging.info(f"PHASE 4: Model Training & Evaluation for task: {task_type.upper()} with {X_data_fs.shape[1]} features.")
    fs_results = {}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data_fs, y_target, test_size=test_size,
        random_state=random_state, stratify=y_target # Stratify by target
    )
    logging.info(f"Data split: Train shape {X_train.shape}, Test shape {X_test.shape}")

    # SMOTE for class balancing (only on training data)
    min_class_count_train = pd.Series(y_train).value_counts().min()
    k_neighbors_smote = max(1, min(smote_k_min, min_class_count_train - 1 if min_class_count_train > 1 else 1))
    
    if task_type == "multiclass" and min_class_count_train <= k_neighbors_smote:
        logging.warning(f"SMOTE k_neighbors ({k_neighbors_smote}) >= min class count ({min_class_count_train}) in multi-class training data. Skipping SMOTE.")
        X_train_bal, y_train_bal = X_train.copy(), y_train.copy()
    elif task_type == "binary" and min_class_count_train <= k_neighbors_smote : # binary can handle k_neighbors = min_class_count -1 even if it's 1.
        logging.warning(f"SMOTE k_neighbors ({k_neighbors_smote}) potentially too high for min class count ({min_class_count_train}) in binary training data. Skipping SMOTE if min_class_count is 1.")
        X_train_bal, y_train_bal = X_train.copy(), y_train.copy()
    else:
        logging.info(f"Applying SMOTE with k_neighbors={k_neighbors_smote}")
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors_smote)
        try:
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            logging.info(f"SMOTE applied. Balanced train shape: {X_train_bal.shape}")
        except ValueError as e:
            logging.warning(f"SMOTE failed: {e}. Using original unbalanced training data.")
            X_train_bal, y_train_bal = X_train.copy(), y_train.copy()

    # Feature Scaling (StandardScaler)
    numerical_cols_train = X_train_bal.select_dtypes(include=np.number).columns.tolist()
    if numerical_cols_train:
        scaler = StandardScaler()
        # Fit on balanced training data's numerical columns
        X_train_bal_scaled = X_train_bal.copy()
        X_train_bal_scaled[numerical_cols_train] = scaler.fit_transform(X_train_bal[numerical_cols_train])
        
        # Transform test data's numerical columns
        X_test_scaled = X_test.copy()
        # Ensure test set has the same numerical columns before transforming
        numerical_cols_test = [col for col in numerical_cols_train if col in X_test.columns]
        if set(numerical_cols_test) != set(numerical_cols_train):
            logging.warning("Mismatch in numerical columns between train_bal and test for scaling. Using common columns.")
        if numerical_cols_test:
             X_test_scaled[numerical_cols_test] = scaler.transform(X_test[numerical_cols_test])
        logging.info("StandardScaler applied to numerical features.")
    else:
        X_train_bal_scaled = X_train_bal.copy()
        X_test_scaled = X_test.copy()
        logging.info("No numerical features found for StandardScaler.")


    for model_name, clf_base_instance in classifiers_dict_config.items():
        clf = clone(clf_base_instance) # Fresh instance for each run
        logging.info(f"Starting {model_name} for {task_type}...")

        try:
            if model_name == "XGBoost" and isinstance(clf, XGBClassifier):
                fit_params = {}
                if task_type == "binary":
                    clf.set_params(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
                else: # multiclass
                    num_unique_classes_in_train_bal = pd.Series(y_train_bal).nunique()
                    clf.set_params(objective='multi:softprob', eval_metric='mlogloss', num_class=num_unique_classes_in_train_bal, use_label_encoder=False)
                logging.info(f"XGBoost params for {task_type}: objective={clf.get_params()['objective']}, eval_metric={clf.get_params()['eval_metric']}, num_class={clf.get_params().get('num_class', 'N/A')}")

            start_train_time = time.time()
            clf.fit(X_train_bal_scaled, y_train_bal)
            train_time = time.time() - start_train_time

            start_pred_time = time.time()
            y_pred = clf.predict(X_test_scaled)
            pred_time = time.time() - start_pred_time
            
            accuracy = accuracy_score(y_test, y_pred)
            # Use target_names for multiclass report if provided
            report_dict = classification_report(y_test, y_pred, target_names=target_names if task_type=="multiclass" else None, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            model_eval_results = {
                "Accuracy": accuracy,
                "Classification Report": report_dict,
                "Confusion Matrix": cm.tolist(),
                "Train Time (s)": round(train_time, 3),
                "Prediction Time (s)": round(pred_time, 3)
            }

            if task_type == "binary":
                # For binary, '1' is usually the positive class
                # Ensure consistent positive label for binary metrics
                pos_label_val = 1 
                model_eval_results["F1-Score"] = f1_score(y_test, y_pred, pos_label=pos_label_val, zero_division=0)
                model_eval_results["Precision"] = precision_score(y_test, y_pred, pos_label=pos_label_val, zero_division=0)
                model_eval_results["Recall"] = recall_score(y_test, y_pred, pos_label=pos_label_val, zero_division=0)
                
                if hasattr(clf, "predict_proba"):
                    y_proba = clf.predict_proba(X_test_scaled)
                    # Check if y_proba is 2D (common case for binary)
                    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                        y_proba_positive = y_proba[:, 1]
                        model_eval_results["AUC-ROC"] = roc_auc_score(y_test, y_proba_positive)
                    # Handle cases where predict_proba might return 1D array or different shape for some classifiers
                    elif y_proba.ndim == 1 : # Can happen if one class is never predicted in train
                         model_eval_results["AUC-ROC"] = roc_auc_score(y_test, y_proba) if pd.Series(y_test).nunique() > 1 else 'N/A (single class in y_test for AUC)'
                    else:
                        model_eval_results["AUC-ROC"] = 'N/A (proba shape issue)'
                else:
                    model_eval_results["AUC-ROC"] = 'N/A (no predict_proba)'
                
                # Specificity (True Negative Rate)
                if cm.size == 4: # Binary confusion matrix
                    tn, fp, fn, tp = cm.ravel()
                    model_eval_results["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
                else: # Should not happen for binary if correctly set up
                    model_eval_results["Specificity"] = 'N/A (CM not 2x2)'
            else: # multi-class
                model_eval_results["Precision (Macro)"] = precision_score(y_test, y_pred, average='macro', zero_division=0)
                model_eval_results["Recall (Macro)"] = recall_score(y_test, y_pred, average='macro', zero_division=0)
                model_eval_results["F1-Score (Macro)"] = f1_score(y_test, y_pred, average='macro', zero_division=0)
                # AUC-ROC for multiclass (OvR or OvO)
                if hasattr(clf, "predict_proba"):
                    y_proba_mc = clf.predict_proba(X_test_scaled)
                    if pd.Series(y_test).nunique() > 1 : # Need at least 2 classes in y_test
                        try:
                           model_eval_results["AUC-ROC (Macro OvR)"] = roc_auc_score(y_test, y_proba_mc, average='macro', multi_class='ovr')
                        except ValueError as auc_e:
                           model_eval_results["AUC-ROC (Macro OvR)"] = f'N/A ({auc_e})'
                    else:
                        model_eval_results["AUC-ROC (Macro OvR)"] = 'N/A (single class in y_test)'
                else:
                    model_eval_results["AUC-ROC (Macro OvR)"] = 'N/A (no predict_proba)'


            fs_results[model_name] = model_eval_results
            logging.info(f"COMPLETED: {model_name} for {task_type}. Accuracy: {accuracy:.4f}, TrainTime: {train_time:.2f}s")

        except Exception as e:
            logging.error(f"Error with {model_name} for {task_type} on {X_data_fs.shape[1]} features: {e}", exc_info=True)
            fs_results[model_name] = {"Error": str(e), "Feature_Count": X_data_fs.shape[1]}
            
    logging.info(f"PHASE 4: Evaluation complete for current feature set for task {task_type.upper()}.")
    return fs_results