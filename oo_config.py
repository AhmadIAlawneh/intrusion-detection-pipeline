import logging
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Removed AdaBoost as it wasn't in your original list of classifiers for training
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

CONFIG = {
    "data_path": "train_test_network (2).csv", # Make sure this file exists in the same directory or provide full path
    "results_dir": "results",
    "random_state": 42,
    "test_size": 0.2,
    "log_level": logging.INFO, # Added for main_pipeline
    "features_to_drop_static": ['src_ip', 'dst_ip', 'src_port', 'dst_port'],
    "correlation_threshold_for_removal": 0.95, # Pearson threshold for redundancy
    "rf_feature_importance_top_n": [10, 15, 20], # List of N for top_N features from each method
    "smote_k_neighbors_multi_min": 2,
    "classifiers": {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='liblinear', n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100),
        "Naive Bayes": GaussianNB(),
        "ANN (MLP)": MLPClassifier(max_iter=300, random_state=42, hidden_layer_sizes=(64,32), early_stopping=True, n_iter_no_change=10),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1), # Default eval_metric, will be overridden in o4
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
        # Add AdaBoost and ExtraTrees here if you want to include them as classifiers:
        # "AdaBoost": AdaBoostClassifier(random_state=42),
        # "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=-1, n_estimators=100),
    }
}