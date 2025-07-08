# o3_feature_selection.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler # For Chi-Square pre-req
from sklearn.feature_selection import chi2, mutual_info_classif
from scipy import stats # For Spearman

def select_features(X, y_binary, corr_threshold, top_n_list, random_state, results_dir):
    logging.info("PHASE 3: Starting feature selection...")
    feature_sets = {}
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Pearson Correlation for Redundancy Removal (as per thesis 3.4.1) ---
    logging.info(f"Applying Pearson Correlation for redundancy removal (threshold: {corr_threshold})...")
    if X.empty:
        logging.error("Input feature matrix X is empty before Pearson correlation.")
        return {}
        
    correlation_matrix = X.corr(method='pearson')
    columns_to_drop_corr = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > corr_threshold:
                colname_to_drop = correlation_matrix.columns[i] # Drop the second one in pair
                columns_to_drop_corr.add(colname_to_drop)

    X_corr_selected = X.drop(columns=list(columns_to_drop_corr), errors='ignore')
    logging.info(f"Pearson correlation filtering: Dropped {len(columns_to_drop_corr)} features. Features remaining: {len(X_corr_selected.columns)}")
    
    if X_corr_selected.empty:
        logging.error("All features dropped after Pearson correlation. Check threshold or data. Cannot proceed with relevance scoring.")
        return {}
    
    # Ensure y_binary is aligned with X_corr_selected's index
    y_binary_aligned = y_binary.loc[X_corr_selected.index]

    # --- Relevance Scoring Methods (Applied to X_corr_selected) ---
    relevance_scores_dfs = {}

    # --- 2.a. Random Forest Importance (Embedded method - thesis 3.4.1) ---
    logging.info("Calculating Random Forest Feature Importances for relevance...")
    # Sample for RF selector training if dataset is large (as in original code)
    sample_size_rf = min(len(X_corr_selected), 75000) 
    if sample_size_rf < len(X_corr_selected):
         X_sample_rf_idx = X_corr_selected.sample(n=sample_size_rf, random_state=random_state).index
         X_sample_rf, y_sample_rf_binary = X_corr_selected.loc[X_sample_rf_idx], y_binary_aligned.loc[X_sample_rf_idx]
    else:
        X_sample_rf, y_sample_rf_binary = X_corr_selected, y_binary_aligned

    rf_selector = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_selector.fit(X_sample_rf, y_sample_rf_binary)
    importances_rf = rf_selector.feature_importances_
    feature_importance_df_rf = pd.DataFrame({'feature': X_corr_selected.columns, 'rf_importance': importances_rf})
    feature_importance_df_rf = feature_importance_df_rf.sort_values(by='rf_importance', ascending=False)
    relevance_scores_dfs['rf'] = feature_importance_df_rf
    feature_importance_df_rf.to_csv(results_path / "rf_feature_importances.csv", index=False)
    logging.info(f"Random Forest importances calculated and saved.")

    # --- 2.b. Spearman Rank Correlation (Filter method - thesis 3.4.1) ---
    logging.info("Calculating Spearman Rank Correlation for relevance...")
    spearman_scores_list = []
    for col in X_corr_selected.columns:
        score, _ = stats.spearmanr(X_corr_selected[col], y_binary_aligned)
        spearman_scores_list.append(abs(score) if not np.isnan(score) else 0) # Use absolute, handle NaN if std dev is 0

    feature_importance_df_spearman = pd.DataFrame({'feature': X_corr_selected.columns, 'spearman_score': spearman_scores_list})
    feature_importance_df_spearman = feature_importance_df_spearman.sort_values(by='spearman_score', ascending=False).fillna(0)
    relevance_scores_dfs['spearman'] = feature_importance_df_spearman
    feature_importance_df_spearman.to_csv(results_path / "spearman_feature_importances.csv", index=False)
    logging.info(f"Spearman Rank correlations calculated and saved.")

    # --- 2.c. Chi-Square Test (Filter method - thesis 3.4.1) ---
    logging.info("Calculating Chi-Square scores for relevance...")
    # Chi2 requires non-negative features. Scale to [0,1] if necessary.
    # OneHotEncoded features are already 0 or 1. Boolean converted are 0 or 1.
    # Other numerical features might be negative or not scaled.
    min_val_check = X_corr_selected.min()
    if (min_val_check < 0).any():
        logging.warning("Some features have negative values. Applying MinMaxScaler before Chi-Square.")
        scaler_chi2 = MinMaxScaler()
        X_chi2_input = scaler_chi2.fit_transform(X_corr_selected)
        X_chi2_input_df = pd.DataFrame(X_chi2_input, columns=X_corr_selected.columns, index=X_corr_selected.index)
    else:
        X_chi2_input_df = X_corr_selected.copy()
    
    # Ensure no NaN values in input to chi2
    X_chi2_input_df.fillna(0, inplace=True)

    chi2_scores_vals, _ = chi2(X_chi2_input_df, y_binary_aligned)
    # Handle potential NaN from chi2 if a feature is constant for all samples of a class
    chi2_scores_vals = np.nan_to_num(chi2_scores_vals, nan=0.0)

    feature_importance_df_chi2 = pd.DataFrame({'feature': X_corr_selected.columns, 'chi2_score': chi2_scores_vals})
    feature_importance_df_chi2 = feature_importance_df_chi2.sort_values(by='chi2_score', ascending=False)
    relevance_scores_dfs['chi2'] = feature_importance_df_chi2
    feature_importance_df_chi2.to_csv(results_path / "chi2_feature_importances.csv", index=False)
    logging.info(f"Chi-Square scores calculated and saved.")

    # --- 2.d. Mutual Information (Filter method - a strong addition) ---
    logging.info("Calculating Mutual Information scores for relevance...")
    mi_scores_vals = mutual_info_classif(X_corr_selected, y_binary_aligned, random_state=random_state)
    feature_importance_df_mi = pd.DataFrame({'feature': X_corr_selected.columns, 'mi_score': mi_scores_vals})
    feature_importance_df_mi = feature_importance_df_mi.sort_values(by='mi_score', ascending=False)
    relevance_scores_dfs['mi'] = feature_importance_df_mi
    feature_importance_df_mi.to_csv(results_path / "mi_feature_importances.csv", index=False)
    logging.info(f"Mutual Information scores calculated and saved.")

    # --- 3. Create Top_N Feature Sets from each relevance scoring method ---
    for method_name, importance_df in relevance_scores_dfs.items():
        score_column_name = importance_df.columns[1] # 'rf_importance', 'spearman_score', etc.
        for n in top_n_list:
            n_actual = min(n, len(importance_df))
            if n_actual == 0:
                logging.warning(f"Cannot select top {n} features for method {method_name} as n_actual is 0.")
                continue
            
            # Select top N features based on the score column
            selected_features_list = importance_df['feature'].head(n_actual).tolist()
            
            fs_key = f'{method_name}_top_{n_actual}_features'
            feature_sets[fs_key] = {
                "features_list": selected_features_list,
                "X_data": X_corr_selected[selected_features_list].copy()
            }
            logging.info(f"Created feature set '{fs_key}' with {len(selected_features_list)} features.")
            
    # --- 4. Add the full set of features after initial Pearson correlation as a baseline ---
    if not X_corr_selected.empty:
        all_features_post_corr_list = X_corr_selected.columns.tolist()
        feature_sets['all_features_post_pearson_corr'] = {
            "features_list": all_features_post_corr_list,
            "X_data": X_corr_selected.copy()
        }
        logging.info(f"Added 'all_features_post_pearson_corr' set with {len(all_features_post_corr_list)} features.")

    if not feature_sets:
        logging.error("No feature sets were created. Check logs for errors in selection methods.")
        
    logging.info(f"PHASE 3: Feature Selection COMPLETE. Total feature sets created: {len(feature_sets)}")
    return feature_sets