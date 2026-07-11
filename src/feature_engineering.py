"""
NovaBank Feature Engineering
=============================
Transforms raw simulator columns into meaningful banking risk features used
for underwriting: affordability ratios, credit-behavior ratios, and
stability indicators.

Usage:
    python feature_engineering.py --in ../data/novabank_customers.csv --out ../data/novabank_features.csv
"""

import argparse
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Affordability ratios -------------------------------------------------
    df["loan_to_income_ratio"] = (
        df["requested_loan_amount"] / (df["monthly_salary"] * 12)
    ).round(3)

    df["emi_to_income_ratio"] = (
        (df["emi"] + df["existing_emi"]) / df["monthly_salary"]
    ).round(3)

    df["total_debt_to_income_ratio"] = (
        (df["emi"] + df["existing_emi"] + df["credit_card_outstanding"] * 0.03)
        / df["monthly_salary"]
    ).round(3)

    df["expense_to_income_ratio"] = (df["monthly_expenses"] / df["monthly_salary"]).round(3)

    df["disposable_income_after_loan"] = (
        df["monthly_salary"] - df["monthly_expenses"] - df["emi"] - df["existing_emi"]
    ).round(0)

    df["savings_to_loan_ratio"] = (
        df["savings_balance"] / df["requested_loan_amount"].replace(0, np.nan)
    ).fillna(0).round(3)

    # --- Credit-behavior ratios -------------------------------------------------
    df["credit_utilization_ratio"] = np.where(
        df["total_credit_limit"] > 0,
        (df["credit_card_outstanding"] / df["total_credit_limit"]).round(3),
        0.0,
    )

    df["debt_burden_score"] = (
        0.5 * df["emi_to_income_ratio"].clip(0, 2)
        + 0.3 * df["credit_utilization_ratio"]
        + 0.2 * (df["existing_loans_count"] / 5).clip(0, 1)
    ).round(3)

    df["delinquency_score"] = (
        df["past_defaults"] * 2 + df["late_payments_last_12m"]
    )

    # --- Stability indicators -------------------------------------------------
    df["income_stability_score"] = (
        (1 - df["income_volatility"]).clip(0, 1) * 0.6
        + (df["years_experience"].clip(0, 20) / 20) * 0.25
        + (df["bank_relationship_years"].clip(0, 20) / 20) * 0.15
    ).round(3)

    df["employment_tenure_ratio"] = np.where(
        df["job_switches"] > 0,
        df["years_experience"] / df["job_switches"].replace(0, 1),
        df["years_experience"],
    ).round(2)

    df["credit_maturity_score"] = (
        np.log1p(df["credit_history_length_years"]) / np.log1p(47)
    ).round(3)

    # --- Composite affordability flag (business rule, not the ML target) ------
    df["loan_amount_to_salary_multiple"] = (
        df["requested_loan_amount"] / df["monthly_salary"]
    ).round(1)

    df["has_collateral_flag"] = df["collateral_available"].astype(int)
    df["has_co_applicant_flag"] = df["co_applicant"].astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(description="NovaBank Feature Engineering")
    parser.add_argument("--in", dest="infile", type=str, default="../data/novabank_customers.csv")
    parser.add_argument("--out", dest="outfile", type=str, default="../data/novabank_features.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.infile)
    feat_df = engineer_features(df)
    feat_df.to_csv(args.outfile, index=False)
    print(f"Engineered features -> {args.outfile}")
    print(f"Shape: {feat_df.shape}")


if __name__ == "__main__":
    main()
