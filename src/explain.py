"""
NovaBank SHAP Explainability
==============================
Loads the trained best model and generates:
  1. A global feature-importance summary plot (which features matter most overall)
  2. Per-customer SHAP explanations (why THIS customer's PD is high/low)

Works with any sklearn Pipeline of the form [preprocessor, classifier], and
supports LogisticRegression, RandomForestClassifier, and XGBClassifier by
picking the right SHAP explainer automatically.

Usage:
    python explain.py --model ../models/best_model.pkl --data ../models/test_split.csv --out-dir ../reports
"""

import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def get_feature_names(preprocessor):
    """Extract output feature names from a fitted ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())


def make_explainer(clf, background_data):
    """Pick the right SHAP explainer for the classifier type."""
    clf_name = type(clf).__name__
    if clf_name == "XGBClassifier":
        return shap.TreeExplainer(clf)
    elif clf_name == "RandomForestClassifier":
        return shap.TreeExplainer(clf)
    elif clf_name == "LogisticRegression":
        return shap.LinearExplainer(clf, background_data)
    else:
        # generic fallback (slower)
        return shap.KernelExplainer(clf.predict_proba, background_data)


def compute_shap_values(pipeline, X_raw, sample_size=500, seed=42):
    """Returns (shap_values_for_class1, feature_names, transformed_X_df, explainer)."""
    preprocessor = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    if len(X_raw) > sample_size:
        X_sample = X_raw.sample(sample_size, random_state=seed)
    else:
        X_sample = X_raw

    X_transformed = preprocessor.transform(X_sample)
    feature_names = get_feature_names(preprocessor)
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X_sample.index)

    explainer = make_explainer(clf, X_transformed_df)
    raw_shap = explainer.shap_values(X_transformed_df)

    # Normalize output across explainer types to a single 2D array for the positive class
    if isinstance(raw_shap, list):
        # [class0_shap, class1_shap]
        shap_values = raw_shap[1]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        # (n_samples, n_features, n_classes)
        shap_values = raw_shap[:, :, 1]
    else:
        shap_values = raw_shap

    return shap_values, feature_names, X_transformed_df, explainer


def plot_global_summary(shap_values, X_transformed_df, out_path, max_display=20):
    plt.figure()
    shap.summary_plot(
        shap_values, X_transformed_df, show=False, max_display=max_display, plot_size=(10, 8)
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_global_bar(shap_values, X_transformed_df, out_path, max_display=15):
    plt.figure()
    shap.summary_plot(
        shap_values, X_transformed_df, plot_type="bar", show=False,
        max_display=max_display, plot_size=(9, 7),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def top_features_for_row(shap_values_row, feature_names, top_n=8):
    """Return top contributing features (sorted by absolute SHAP value) for one prediction."""
    order = np.argsort(-np.abs(shap_values_row))[:top_n]
    return [
        {"feature": feature_names[i], "shap_value": round(float(shap_values_row[i]), 4)}
        for i in order
    ]


def global_feature_importance(shap_values, feature_names, top_n=15):
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_n]
    return [
        {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in order
    ]


def main():
    parser = argparse.ArgumentParser(description="NovaBank SHAP Explainability")
    parser.add_argument("--model", type=str, default="../models/best_model.pkl")
    parser.add_argument("--data", type=str, default="../models/test_split.csv")
    parser.add_argument("--out-dir", type=str, default="../reports")
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    df = pd.read_csv(args.data)
    y = df["default"] if "default" in df.columns else None
    X_raw = df.drop(columns=["default"]) if "default" in df.columns else df

    shap_values, feature_names, X_transformed_df, explainer = compute_shap_values(
        pipeline, X_raw, sample_size=args.sample_size
    )

    plot_global_summary(shap_values, X_transformed_df, f"{args.out_dir}/shap_summary_beeswarm.png")
    plot_global_bar(shap_values, X_transformed_df, f"{args.out_dir}/shap_summary_bar.png")

    global_importance = global_feature_importance(shap_values, feature_names)
    with open(f"{args.out_dir}/global_feature_importance.json", "w") as f:
        json.dump(global_importance, f, indent=2)

    print("Top global risk drivers (mean |SHAP value|):")
    for item in global_importance[:10]:
        print(f"  {item['feature']:35s} {item['mean_abs_shap']}")

    print(f"\nSaved beeswarm plot -> {args.out_dir}/shap_summary_beeswarm.png")
    print(f"Saved bar plot -> {args.out_dir}/shap_summary_bar.png")
    print(f"Saved global importance json -> {args.out_dir}/global_feature_importance.json")


if __name__ == "__main__":
    main()
