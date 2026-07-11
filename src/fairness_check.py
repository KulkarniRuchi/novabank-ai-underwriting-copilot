"""
NovaBank Fairness & Bias Analysis
====================================
Checks whether the trained PD model treats demographic groups consistently.
This matters for lending models specifically: regulators (and good practice)
expect banks to monitor for disparate impact — where a protected or
demographic group is approved/declined at meaningfully different rates than
others, even if no single feature explicitly encodes that group.

For each group (e.g. each gender, each city tier) on the held-out test set,
this computes:
  - n                  : group size
  - actual_default_rate: true default rate in that group (ground truth)
  - avg_predicted_pd   : average model-predicted PD in that group
  - approval_rate      : share of applicants the POLICY would Approve (PD < 0.08)
  - decline_rate       : share the POLICY would Decline (PD >= 0.22)
  - fpr                : false positive rate at PD>=0.5 (good payers wrongly
                          flagged as high-risk) — an "equalized odds" check
  - tpr_recall         : true positive rate at PD>=0.5 (actual defaulters
                          correctly caught) — do we protect all groups equally?

It also applies the "four-fifths rule" (a common, if crude, US regulatory
rule of thumb from EEOC/fair-lending practice): if the lowest approval rate
across groups is under 80% of the highest approval rate, that's flagged as
a disparate-impact concern worth a closer look — NOT an automatic verdict
of discrimination, since the underlying risk factors may differ legitimately
across groups. It's a screening signal, not a conclusion.

Usage:
    python fairness_check.py --model ../models/best_model.pkl --data ../models/test_split.csv --out-dir ../reports
"""

import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

APPROVE_THRESHOLD = 0.08   # matches the business policy used elsewhere in the app
DECLINE_THRESHOLD = 0.22
CLASSIFICATION_THRESHOLD = 0.5  # standard threshold for FPR/recall, matches train_model.py

FOUR_FIFTHS_RULE_CUTOFF = 0.8


def compute_group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_val, g in df.groupby(group_col):
        n = len(g)
        actual_default_rate = g["default"].mean()
        avg_pd = g["predicted_pd"].mean()
        approval_rate = (g["predicted_pd"] < APPROVE_THRESHOLD).mean()
        decline_rate = (g["predicted_pd"] >= DECLINE_THRESHOLD).mean()

        predicted_positive = g["predicted_pd"] >= CLASSIFICATION_THRESHOLD  # "predicted will default"
        actual_positive = g["default"] == 1
        actual_negative = g["default"] == 0

        # False Positive Rate: among actual GOOD payers, how many did we wrongly flag as high risk?
        fpr = (predicted_positive & actual_negative).sum() / max(actual_negative.sum(), 1)
        # Recall / TPR: among actual defaulters, how many did we correctly catch?
        tpr_recall = (predicted_positive & actual_positive).sum() / max(actual_positive.sum(), 1)

        rows.append({
            group_col: group_val,
            "n": n,
            "actual_default_rate": round(float(actual_default_rate), 4),
            "avg_predicted_pd": round(float(avg_pd), 4),
            "approval_rate": round(float(approval_rate), 4),
            "decline_rate": round(float(decline_rate), 4),
            "fpr": round(float(fpr), 4),
            "tpr_recall": round(float(tpr_recall), 4),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def four_fifths_rule_check(metrics_df: pd.DataFrame, group_col: str, min_group_size: int = 30) -> dict:
    """
    Applies the four-fifths rule to approval rates. Small groups (below
    min_group_size) are excluded from the ratio calculation since their rates
    are too noisy to be meaningful, but are still reported in the raw table.
    """
    reliable = metrics_df[metrics_df["n"] >= min_group_size]
    if len(reliable) < 2:
        return {"applicable": False, "reason": "Not enough groups with sufficient sample size."}

    max_rate = reliable["approval_rate"].max()
    min_rate = reliable["approval_rate"].min()
    ratio = min_rate / max_rate if max_rate > 0 else 1.0

    worst_group = reliable.loc[reliable["approval_rate"].idxmin(), group_col]
    best_group = reliable.loc[reliable["approval_rate"].idxmax(), group_col]

    return {
        "applicable": True,
        "impact_ratio": round(float(ratio), 4),
        "passes_four_fifths_rule": bool(ratio >= FOUR_FIFTHS_RULE_CUTOFF),
        "lowest_approval_group": str(worst_group),
        "lowest_approval_rate": round(float(min_rate), 4),
        "highest_approval_group": str(best_group),
        "highest_approval_rate": round(float(max_rate), 4),
    }


def plot_approval_rates(metrics_df: pd.DataFrame, group_col: str, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#d62728" if v < FOUR_FIFTHS_RULE_CUTOFF * metrics_df["approval_rate"].max() else "#2ca02c"
              for v in metrics_df["approval_rate"]]
    ax.bar(metrics_df[group_col].astype(str), metrics_df["approval_rate"], color=colors)
    ax.set_ylabel("Approval rate (PD < 8%)")
    ax.set_title(title)
    ax.set_ylim(0, max(metrics_df["approval_rate"].max() * 1.3, 0.05))
    for i, v in enumerate(metrics_df["approval_rate"]):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_fairness_analysis(model_path: str, data_path: str, out_dir: str, group_cols=("gender", "city_tier")):
    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

    if "default" not in df.columns:
        raise ValueError("Expected a 'default' column (ground truth) in the input data for fairness analysis.")

    X = df.drop(columns=["default"])
    df = df.copy()
    df["predicted_pd"] = pipeline.predict_proba(X)[:, 1]

    all_results = {}
    for group_col in group_cols:
        if group_col not in df.columns:
            print(f"Skipping '{group_col}' — not found in data.")
            continue

        metrics_df = compute_group_metrics(df, group_col)
        rule_check = four_fifths_rule_check(metrics_df, group_col)

        csv_path = f"{out_dir}/fairness_{group_col}.csv"
        metrics_df.to_csv(csv_path, index=False)

        plot_path = f"{out_dir}/fairness_approval_rate_by_{group_col}.png"
        plot_approval_rates(metrics_df, group_col, plot_path, f"Approval Rate by {group_col.replace('_', ' ').title()}")

        print(f"\n=== Fairness check: {group_col} ===")
        print(metrics_df.to_string(index=False))
        print(f"\nFour-fifths rule: {rule_check}")

        all_results[group_col] = {
            "metrics": metrics_df.to_dict(orient="records"),
            "four_fifths_rule": rule_check,
        }

    with open(f"{out_dir}/fairness_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved per-group CSVs, bar charts, and fairness_summary.json -> {out_dir}/")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="NovaBank Fairness & Bias Analysis")
    parser.add_argument("--model", type=str, default="../models/best_model.pkl")
    parser.add_argument("--data", type=str, default="../models/test_split.csv")
    parser.add_argument("--out-dir", type=str, default="../reports")
    parser.add_argument("--groups", nargs="+", default=["gender", "city_tier"])
    args = parser.parse_args()

    run_fairness_analysis(args.model, args.data, args.out_dir, group_cols=tuple(args.groups))


if __name__ == "__main__":
    main()
