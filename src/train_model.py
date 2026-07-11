"""
NovaBank Loan Default Prediction Model
========================================
Trains and compares Logistic Regression, Random Forest, and XGBoost models
to predict Probability of Default (PD). Saves the best model (by ROC-AUC)
along with the preprocessing pipeline for downstream use by the SHAP
explainer and the AI Underwriting Copilot.

Usage:
    python train_model.py --in ../data/novabank_features.csv --model-dir ../models
"""

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# Columns that should never be used as model inputs (identifiers / leakage / target)
DROP_COLS = ["customer_id", "true_default_probability", "default"]

CATEGORICAL_COLS = [
    "gender", "city_tier", "marital_status", "education", "occupation",
    "employer_type", "loan_purpose",
]

BOOLEAN_COLS = ["collateral_available", "co_applicant"]


def load_data(path):
    df = pd.read_csv(path)
    return df


def build_preprocessor(df, categorical_cols, boolean_cols, numeric_cols):
    for c in boolean_cols:
        df[c] = df[c].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="passthrough",  # boolean_cols already converted to int, pass through
    )
    return preprocessor


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Train NovaBank PD models")
    parser.add_argument("--in", dest="infile", type=str, default="../data/novabank_features.csv")
    parser.add_argument("--model-dir", type=str, default="../models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.infile)

    y = df["default"].astype(int)
    X_raw = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    boolean_cols = [c for c in BOOLEAN_COLS if c in X_raw.columns]
    for c in boolean_cols:
        X_raw[c] = X_raw[c].astype(int)
    categorical_cols = [c for c in CATEGORICAL_COLS if c in X_raw.columns]
    numeric_cols = [c for c in X_raw.columns if c not in categorical_cols and c not in boolean_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols + boolean_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
    )

    results = {}
    fitted_models = {}

    # ---------------- Logistic Regression ----------------
    lr_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
    ])
    lr_pipe.fit(X_train, y_train)
    results["logistic_regression"] = evaluate(lr_pipe, X_test, y_test)
    fitted_models["logistic_regression"] = lr_pipe

    # ---------------- Random Forest ----------------
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=15,
            class_weight="balanced", random_state=args.seed, n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    results["random_forest"] = evaluate(rf_pipe, X_test, y_test)
    fitted_models["random_forest"] = rf_pipe

    # ---------------- XGBoost ----------------
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", xgb.XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc", random_state=args.seed, n_jobs=-1,
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    results["xgboost"] = evaluate(xgb_pipe, X_test, y_test)
    fitted_models["xgboost"] = xgb_pipe

    # ---------------- Pick best model by ROC-AUC ----------------
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_name]

    print("Model comparison (ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1):")
    for name, m in results.items():
        marker = "  <-- BEST" if name == best_name else ""
        print(f"  {name:20s} {m}{marker}")

    # ---------------- Save artifacts ----------------
    joblib.dump(best_model, f"{args.model_dir}/best_model.pkl")
    joblib.dump(fitted_models, f"{args.model_dir}/all_models.pkl")

    metadata = {
        "best_model": best_name,
        "results": results,
        "numeric_cols": numeric_cols,
        "boolean_cols": boolean_cols,
        "categorical_cols": categorical_cols,
        "feature_cols": list(X_raw.columns),
        "test_size": args.test_size,
        "seed": args.seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "default_rate_train": round(float(y_train.mean()), 4),
        "default_rate_test": round(float(y_test.mean()), 4),
    }
    with open(f"{args.model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # save the test split for downstream SHAP / demo use
    X_test.assign(default=y_test.values).to_csv(f"{args.model_dir}/test_split.csv", index=False)
    X_train.assign(default=y_train.values).to_csv(f"{args.model_dir}/train_split.csv", index=False)

    print(f"\nBest model: {best_name}")
    print(f"Saved best model -> {args.model_dir}/best_model.pkl")
    print(f"Saved metadata -> {args.model_dir}/metadata.json")


if __name__ == "__main__":
    main()
