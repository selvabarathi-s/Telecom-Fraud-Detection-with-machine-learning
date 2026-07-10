import argparse
import pickle
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

RANDOM_STATE = 42
TARGET_COL = "fraud_label"
ID_COL = "call_id"


def load_and_clean(path: str) -> pd.DataFrame:
    dataset = pd.read_csv(path)

    if ID_COL in dataset.columns:
        dataset = dataset.drop(columns=[ID_COL])

    # drop duplicate rows
    dataset = dataset.drop_duplicates()

    # drop rows where the target itself is missing
    dataset = dataset.dropna(subset=[TARGET_COL])
    dataset[TARGET_COL] = dataset[TARGET_COL].astype(int)

    feature_cols = dataset.columns.drop(TARGET_COL)

    # impute missing feature values with the column median
    for col in feature_cols:
        if dataset[col].isnull().sum() > 0:
            dataset[col] = dataset[col].fillna(dataset[col].median())

    return dataset, list(feature_cols)


def cap_outliers(dataset: pd.DataFrame, feature_cols):
    """Cap outliers to IQR bounds (winsorizing). Returns clip bounds so the
    app can apply/display the same bounds for live user input."""
    bounds = {}
    for col in feature_cols:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        dataset[col] = dataset[col].clip(lower, upper)
        bounds[col] = {"lower": float(lower), "upper": float(upper)}
    return dataset, bounds


def evaluate_model(name, y_actual, y_pred, y_proba=None):
    metrics = {
        "Model": name,
        "Accuracy": float(accuracy_score(y_actual, y_pred)),
        "Precision": float(precision_score(y_actual, y_pred)),
        "Recall": float(recall_score(y_actual, y_pred)),
        "F1": float(f1_score(y_actual, y_pred)),
    }
    if y_proba is not None:
        metrics["ROC_AUC"] = float(roc_auc_score(y_actual, y_proba))
    cm = confusion_matrix(y_actual, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = classification_report(
        y_actual, y_pred, target_names=["Not Fraud", "Fraud"]
    )
    return metrics


def main(data_path: str):
    print(f"Loading data from: {data_path}")
    dataset, feature_cols = load_and_clean(data_path)
    print(f"Shape after cleaning: {dataset.shape}")

    # Store raw (pre-outlier-cap) min/max per feature -> used to build
    # sensible, permissive input slider ranges in the Streamlit app.
    raw_min_max = {
        col: {"min": float(dataset[col].min()), "max": float(dataset[col].max())}
        for col in feature_cols
    }

    dataset, clip_bounds = cap_outliers(dataset, feature_cols)

    medians = {col: float(dataset[col].median()) for col in feature_cols}

    X = dataset[feature_cols]
    y = dataset[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE, test_size=0.2, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    results = []
    model_lookup = {}

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    rf_proba = rf.predict_proba(x_test)[:, 1]
    results.append(evaluate_model("Random Forest", y_test, rf_pred, rf_proba))
    model_lookup["Random Forest"] = rf

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="f1")
    print("Random Forest CV F1:", cv_scores.mean(), "+/-", cv_scores.std())

    # --- Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    log_reg.fit(x_train_scaled, y_train)
    lr_pred = log_reg.predict(x_test_scaled)
    lr_proba = log_reg.predict_proba(x_test_scaled)[:, 1]
    results.append(evaluate_model("Logistic Regression", y_test, lr_pred, lr_proba))
    model_lookup["Logistic Regression"] = log_reg

    # --- Decision Tree ---
    tree = DecisionTreeClassifier(
        criterion="gini", max_depth=8, min_samples_split=2, min_samples_leaf=1,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    tree.fit(x_train, y_train)
    tree_pred = tree.predict(x_test)
    tree_proba = tree.predict_proba(x_test)[:, 1]
    results.append(evaluate_model("Decision Tree", y_test, tree_pred, tree_proba))
    model_lookup["Decision Tree"] = tree

    # --- SVM ---
    svm_model = SVC(
        C=1.0, kernel="rbf", gamma="scale", probability=True,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    svm_model.fit(x_train_scaled, y_train)
    svm_pred = svm_model.predict(x_test_scaled)
    svm_proba = svm_model.predict_proba(x_test_scaled)[:, 1]
    results.append(evaluate_model("SVM (RBF)", y_test, svm_pred, svm_proba))
    model_lookup["SVM (RBF)"] = svm_model

    # --- KNN ---
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski", n_jobs=-1)
    knn.fit(x_train_scaled, y_train)
    knn_pred = knn.predict(x_test_scaled)
    knn_proba = knn.predict_proba(x_test_scaled)[:, 1]
    results.append(evaluate_model("KNN", y_test, knn_pred, knn_proba))
    model_lookup["KNN"] = knn

    results_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("confusion_matrix", "classification_report")} for r in results]
    ).sort_values("F1", ascending=False).reset_index(drop=True)
    print(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = model_lookup[best_model_name]
    needs_scaling = best_model_name in ("Logistic Regression", "SVM (RBF)", "KNN")
    print("Best model:", best_model_name)

    # Feature importance (only meaningful for tree-based models)
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(zip(feature_cols, best_model.feature_importances_.tolist()))
    elif best_model_name == "Logistic Regression":
        feature_importance = dict(zip(feature_cols, np.abs(best_model.coef_[0]).tolist()))

    # Random Forest feature importance always saved too (for the "why" chart)
    rf_importance = dict(zip(feature_cols, rf.feature_importances_.tolist()))

    artifact = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "needs_scaling": needs_scaling,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "medians": medians,
        "clip_bounds": clip_bounds,
        "raw_min_max": raw_min_max,
        "feature_importance": feature_importance,
        "rf_feature_importance": rf_importance,
        "results": results,
        "results_df": results_df,
        "correlation_matrix": dataset[feature_cols + [TARGET_COL]].corr(),
        "class_balance": y.value_counts(normalize=True).to_dict(),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "n_rows": int(dataset.shape[0]),
    }

    with open("model_artifact.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print("\nSaved model_artifact.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str,
        default="telecom_fraud_detection_dataset_impure.csv",
        help="Path to the raw CSV dataset"
    )
    args = parser.parse_args()
    main(args.data)
