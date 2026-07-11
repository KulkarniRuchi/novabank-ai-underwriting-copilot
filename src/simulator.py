"""
NovaBank Customer Simulator
============================
Generates realistic synthetic Indian retail-banking customers using a
domain-driven, business-rule based dependency chain:

    Demographics -> Career -> Finance -> Credit History -> Loan Application
    -> Risk Score -> Default

Each stage is a function of the previous stages, so the resulting dataset
has realistic internal correlations (e.g. higher education -> higher salary
-> higher loan eligibility -> lower default risk, etc.) instead of being
independently randomized like a naive synthetic dataset.

Usage:
    python simulator.py --n 10000 --out ../data/novabank_customers.csv
"""

import argparse
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker("en_IN")

RNG_SEED = 42

CITY_TIERS = ["Tier-1", "Tier-2", "Tier-3"]
CITY_TIER_WEIGHTS = [0.35, 0.40, 0.25]
CITY_TIER_COST_INDEX = {"Tier-1": 1.35, "Tier-2": 1.05, "Tier-3": 0.80}

EDUCATION_LEVELS = ["High School", "Diploma", "Bachelor's", "Master's", "PhD/Professional"]
EDUCATION_WEIGHTS = [0.12, 0.18, 0.45, 0.20, 0.05]
# multiplier applied to base income potential
EDUCATION_INCOME_FACTOR = {
    "High School": 0.55,
    "Diploma": 0.75,
    "Bachelor's": 1.00,
    "Master's": 1.35,
    "PhD/Professional": 1.70,
}

OCCUPATIONS = [
    "Salaried - Private", "Salaried - Government", "Salaried - PSU",
    "Self-Employed - Professional", "Self-Employed - Business",
    "Gig/Freelance", "Unemployed/Student",
]
OCCUPATION_WEIGHTS = [0.38, 0.14, 0.08, 0.10, 0.14, 0.10, 0.06]

OCCUPATION_INCOME_FACTOR = {
    "Salaried - Private": 1.00,
    "Salaried - Government": 0.85,
    "Salaried - PSU": 0.95,
    "Self-Employed - Professional": 1.30,
    "Self-Employed - Business": 1.15,
    "Gig/Freelance": 0.70,
    "Unemployed/Student": 0.15,
}

# income stability: how volatile month-to-month income is (0=very stable,1=very volatile)
OCCUPATION_STABILITY_BASE = {
    "Salaried - Private": 0.15,
    "Salaried - Government": 0.05,
    "Salaried - PSU": 0.08,
    "Self-Employed - Professional": 0.35,
    "Self-Employed - Business": 0.45,
    "Gig/Freelance": 0.60,
    "Unemployed/Student": 0.80,
}

LOAN_PURPOSES = ["Home", "Vehicle", "Personal", "Education", "Business Expansion", "Gold/Consumer Durable"]

EMPLOYER_TYPES = ["MNC", "Large Domestic Co.", "SME", "Startup", "Government", "Self-Employed", "None"]


def _rng(seed=None):
    return np.random.default_rng(seed if seed is not None else RNG_SEED)


# ---------------------------------------------------------------------------
# Stage 1: Demographics
# ---------------------------------------------------------------------------
def simulate_demographics(n, rng):
    age = np.clip(rng.gamma(shape=3.2, scale=5.2, size=n) + 21, 21, 65).astype(int)
    gender = rng.choice(["Male", "Female", "Other"], size=n, p=[0.60, 0.38, 0.02])
    city_tier = rng.choice(CITY_TIERS, size=n, p=CITY_TIER_WEIGHTS)
    marital_status = np.where(
        age < 25,
        rng.choice(["Single", "Married"], size=n, p=[0.85, 0.15]),
        rng.choice(["Single", "Married", "Divorced", "Widowed"], size=n, p=[0.25, 0.62, 0.09, 0.04]),
    )
    dependents = np.where(
        marital_status == "Married",
        rng.poisson(1.4, size=n),
        rng.poisson(0.3, size=n),
    )
    dependents = np.clip(dependents, 0, 6)

    education = rng.choice(EDUCATION_LEVELS, size=n, p=EDUCATION_WEIGHTS)
    # very young people are unlikely to already hold Master's/PhD -> lightly resample
    too_young_for_advanced = (age < 24) & (np.isin(education, ["Master's", "PhD/Professional"]))
    if too_young_for_advanced.any():
        education = education.copy()
        education[too_young_for_advanced] = rng.choice(
            ["High School", "Diploma", "Bachelor's"], size=too_young_for_advanced.sum(), p=[0.25, 0.35, 0.40]
        )

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "city_tier": city_tier,
        "marital_status": marital_status,
        "dependents": dependents,
        "education": education,
    })
    return df


# ---------------------------------------------------------------------------
# Stage 2: Career
# ---------------------------------------------------------------------------
def simulate_career(df, rng):
    n = len(df)
    age = df["age"].values
    education = df["education"].values

    working_age = np.clip(age - rng.integers(18, 24, size=n), 0, None)

    occ_probs = np.tile(OCCUPATION_WEIGHTS, (n, 1))
    # young people more likely unemployed/student or gig; skew probabilities by age
    young_mask = age < 23
    occ_probs[young_mask, OCCUPATIONS.index("Unemployed/Student")] *= 4
    occ_probs[young_mask] = occ_probs[young_mask] / occ_probs[young_mask].sum(axis=1, keepdims=True)
    occ_probs = occ_probs / occ_probs.sum(axis=1, keepdims=True)

    occupation = np.array([
        rng.choice(OCCUPATIONS, p=occ_probs[i]) for i in range(n)
    ])

    years_experience = np.where(
        occupation == "Unemployed/Student",
        0,
        np.clip(working_age - rng.integers(0, 3, size=n), 0, 40),
    )

    employer_type = []
    for occ in occupation:
        if occ.startswith("Salaried - Government"):
            employer_type.append("Government")
        elif occ.startswith("Salaried"):
            employer_type.append(rng.choice(["MNC", "Large Domestic Co.", "SME", "Startup"], p=[0.30, 0.30, 0.30, 0.10]))
        elif occ.startswith("Self-Employed"):
            employer_type.append("Self-Employed")
        elif occ == "Unemployed/Student":
            employer_type.append("Not Employed")
        else:
            employer_type.append(rng.choice(["SME", "Startup", "Not Applicable"], p=[0.4, 0.3, 0.3]))
    employer_type = np.array(employer_type)

    job_switches = np.clip(
        (years_experience / rng.uniform(2.5, 5.0, size=n)).astype(int) + rng.integers(-1, 2, size=n), 0, 12
    )

    df = df.copy()
    df["occupation"] = occupation
    df["years_experience"] = years_experience
    df["employer_type"] = employer_type
    df["job_switches"] = job_switches
    return df


# ---------------------------------------------------------------------------
# Stage 3: Finance
# ---------------------------------------------------------------------------
def simulate_finance(df, rng):
    n = len(df)

    base_income = 22000  # base monthly salary anchor in INR
    edu_factor = df["education"].map(EDUCATION_INCOME_FACTOR).values
    occ_factor = df["occupation"].map(OCCUPATION_INCOME_FACTOR).values
    city_factor = df["city_tier"].map(CITY_TIER_COST_INDEX).values
    exp_factor = 1 + np.log1p(df["years_experience"].values) * 0.28

    income_noise = rng.lognormal(mean=0.0, sigma=0.25, size=n)

    monthly_salary = (
        base_income * edu_factor * occ_factor * city_factor * exp_factor * income_noise
    )
    monthly_salary = np.clip(monthly_salary, 8000, 1_200_000).round(-2)

    stability_base = df["occupation"].map(OCCUPATION_STABILITY_BASE).values
    income_volatility = np.clip(stability_base + rng.normal(0, 0.06, size=n), 0.02, 0.95)

    # monthly expenses driven by dependents, city cost, lifestyle noise
    dependents = df["dependents"].values
    expense_ratio = np.clip(
        0.35 + 0.05 * dependents + rng.normal(0, 0.08, size=n) - 0.05 * (city_factor - 1),
        0.20, 0.90,
    )
    monthly_expenses = (monthly_salary * expense_ratio).round(-2)

    disposable_income = np.clip(monthly_salary - monthly_expenses, 500, None)

    # savings accumulate over years of experience, minus volatility drag
    years_saving = np.clip(df["years_experience"].values, 0, 35)
    savings_rate = np.clip(0.15 - income_volatility * 0.10 + rng.normal(0, 0.03, size=n), 0.02, 0.35)
    savings_balance = (
        disposable_income * savings_rate * 12 * years_saving * rng.uniform(0.5, 1.1, size=n)
    ).round(-2)
    savings_balance = np.clip(savings_balance, 0, None)

    existing_investments = (savings_balance * rng.uniform(0.0, 0.6, size=n)).round(-2)

    bank_relationship_years = np.clip(
        (df["age"].values - rng.integers(18, 22, size=n)).astype(int) - rng.integers(0, 5, size=n), 0, None
    )

    df = df.copy()
    df["monthly_salary"] = monthly_salary
    df["monthly_expenses"] = monthly_expenses
    df["disposable_income"] = disposable_income
    df["income_volatility"] = income_volatility.round(3)
    df["savings_balance"] = savings_balance
    df["existing_investments"] = existing_investments
    df["bank_relationship_years"] = bank_relationship_years
    return df


# ---------------------------------------------------------------------------
# Stage 4: Credit History
# ---------------------------------------------------------------------------
def simulate_credit_history(df, rng):
    n = len(df)

    # base credit score influenced by income stability, experience, existing savings
    stability_bonus = (1 - df["income_volatility"].values) * 80
    experience_bonus = np.clip(df["years_experience"].values, 0, 15) * 3
    savings_bonus = np.clip(np.log1p(df["savings_balance"].values) * 4.5, 0, 60)
    noise = rng.normal(0, 55, size=n)

    credit_score = 480 + stability_bonus + experience_bonus + savings_bonus + noise
    credit_score = np.clip(credit_score, 300, 900).round().astype(int)

    credit_history_length = np.clip(
        df["bank_relationship_years"].values - rng.integers(0, 3, size=n), 0, None
    )

    num_credit_cards = np.clip(
        rng.poisson(1.2 + credit_score / 900 * 2.0, size=n) - (df["occupation"].values == "Unemployed/Student").astype(int),
        0, 8,
    )

    total_credit_limit = (num_credit_cards * (credit_score / 900) * rng.uniform(30000, 120000, size=n)).round(-3)
    total_credit_limit = np.clip(total_credit_limit, 0, None)

    # utilization tends to be higher for lower credit scores / higher volatility
    utilization_base = np.clip(
        0.55 - (credit_score - 300) / 600 * 0.35 + df["income_volatility"].values * 0.25 + rng.normal(0, 0.08, size=n),
        0.0, 0.98,
    )
    credit_card_outstanding = np.where(
        total_credit_limit > 0, (total_credit_limit * utilization_base).round(-2), 0
    )

    existing_loans_count = np.clip(
        rng.poisson(0.6 + df["years_experience"].values / 25, size=n), 0, 5
    )
    existing_emi = (
        existing_loans_count * df["monthly_salary"].values * rng.uniform(0.03, 0.10, size=n)
    ).round(-2)

    # probability of past default/delinquency events, worse for low score & high volatility
    past_default_prob = np.clip(
        0.25 - (credit_score - 300) / 600 * 0.22 + df["income_volatility"].values * 0.18, 0.01, 0.6
    )
    past_defaults = rng.binomial(3, past_default_prob)
    late_payments_last_12m = np.clip(
        rng.poisson(past_default_prob * 6, size=n), 0, 12
    )

    df = df.copy()
    df["credit_score"] = credit_score
    df["credit_history_length_years"] = credit_history_length
    df["num_credit_cards"] = num_credit_cards
    df["total_credit_limit"] = total_credit_limit
    df["credit_card_outstanding"] = credit_card_outstanding
    df["credit_utilization_ratio"] = np.where(
        total_credit_limit > 0, (credit_card_outstanding / np.clip(total_credit_limit, 1, None)).round(3), 0.0
    )
    df["existing_loans_count"] = existing_loans_count
    df["existing_emi"] = existing_emi
    df["past_defaults"] = past_defaults
    df["late_payments_last_12m"] = late_payments_last_12m
    return df


# ---------------------------------------------------------------------------
# Stage 5: Loan Application
# ---------------------------------------------------------------------------
def simulate_loan_application(df, rng):
    n = len(df)

    loan_purpose = rng.choice(LOAN_PURPOSES, size=n, p=[0.28, 0.20, 0.22, 0.10, 0.12, 0.08])

    purpose_multiplier = {
        "Home": 45, "Vehicle": 10, "Personal": 4.5, "Education": 8,
        "Business Expansion": 12, "Gold/Consumer Durable": 2.5,
    }
    mult = np.array([purpose_multiplier[p] for p in loan_purpose])

    requested_loan_amount = (
        df["monthly_salary"].values * mult * rng.uniform(0.55, 1.3, size=n)
    ).round(-3)
    requested_loan_amount = np.clip(requested_loan_amount, 20000, 15_000_000)

    tenure_choices = {
        "Home": [120, 180, 240, 300], "Vehicle": [36, 48, 60, 84],
        "Personal": [12, 24, 36, 60], "Education": [36, 60, 84, 120],
        "Business Expansion": [24, 36, 60, 84], "Gold/Consumer Durable": [6, 12, 18, 24],
    }
    loan_tenure_months = np.array([
        rng.choice(tenure_choices[p]) for p in loan_purpose
    ])

    base_rate = {
        "Home": 8.5, "Vehicle": 9.5, "Personal": 13.0, "Education": 10.5,
        "Business Expansion": 12.5, "Gold/Consumer Durable": 14.0,
    }
    credit_score = df["credit_score"].values
    risk_premium = np.clip((750 - credit_score) / 100, -1.0, 5.0)
    interest_rate = np.array([base_rate[p] for p in loan_purpose]) + risk_premium + rng.normal(0, 0.4, size=n)
    interest_rate = np.clip(interest_rate, 6.5, 24.0).round(2)

    r = interest_rate / 1200
    t = loan_tenure_months
    emi = np.where(
        r > 0,
        requested_loan_amount * r * (1 + r) ** t / (((1 + r) ** t) - 1),
        requested_loan_amount / t,
    ).round(0)

    collateral_available = rng.choice(
        [True, False], size=n,
        p=[0.55, 0.45],
    )
    # home/vehicle loans are almost always collateralized by the asset itself
    collateral_available = np.where(
        np.isin(loan_purpose, ["Home", "Vehicle"]), True, collateral_available
    )

    co_applicant = rng.choice([True, False], size=n, p=[0.35, 0.65])

    df = df.copy()
    df["loan_purpose"] = loan_purpose
    df["requested_loan_amount"] = requested_loan_amount
    df["loan_tenure_months"] = loan_tenure_months
    df["interest_rate_pct"] = interest_rate
    df["emi"] = emi
    df["collateral_available"] = collateral_available
    df["co_applicant"] = co_applicant
    return df


# ---------------------------------------------------------------------------
# Stage 6: Risk Score & Default
# ---------------------------------------------------------------------------
def simulate_risk_and_default(df, rng, target_default_rate=0.09):
    n = len(df)

    salary = df["monthly_salary"].values
    total_emi_burden = df["emi"].values + df["existing_emi"].values
    emi_to_income = total_emi_burden / np.clip(salary, 1, None)
    loan_to_income_annual = df["requested_loan_amount"].values / np.clip(salary * 12, 1, None)

    z = (
        -0.014 * (df["credit_score"].values - 650)
        + 4.2 * np.clip(emi_to_income, 0, 3)
        + 0.55 * loan_to_income_annual
        + 3.0 * df["credit_utilization_ratio"].values
        + 0.55 * df["past_defaults"].values
        + 0.16 * df["late_payments_last_12m"].values
        + 2.0 * df["income_volatility"].values
        + 0.10 * df["dependents"].values
        - 0.35 * np.log1p(df["years_experience"].values)
        - 0.30 * np.log1p(df["bank_relationship_years"].values)
        - 0.9 * df["collateral_available"].values.astype(float)
        - 0.5 * df["co_applicant"].values.astype(float)
        - 0.20 * (df["employer_type"].values == "Government").astype(float)
        + 0.35 * (df["occupation"].values == "Unemployed/Student").astype(float)
        + rng.normal(0, 0.9, size=n)
    )

    # calibrate intercept so overall default rate lands near target
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = sigmoid(z + mid).mean()
        if rate > target_default_rate:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2

    pd_true = sigmoid(z + intercept)
    default = rng.binomial(1, pd_true)

    df = df.copy()
    df["emi_to_income_ratio"] = emi_to_income.round(3)
    df["loan_to_income_ratio"] = loan_to_income_annual.round(3)
    df["true_default_probability"] = pd_true.round(4)
    df["default"] = default
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_customers(n=10000, seed=RNG_SEED, target_default_rate=0.09):
    rng = _rng(seed)
    df = simulate_demographics(n, rng)
    df = simulate_career(df, rng)
    df = simulate_finance(df, rng)
    df = simulate_credit_history(df, rng)
    df = simulate_loan_application(df, rng)
    df = simulate_risk_and_default(df, rng, target_default_rate=target_default_rate)

    df.insert(0, "customer_id", [f"NB{100000+i}" for i in range(len(df))])
    return df


def main():
    parser = argparse.ArgumentParser(description="NovaBank Customer Simulator")
    parser.add_argument("--n", type=int, default=10000, help="number of customers to generate")
    parser.add_argument("--out", type=str, default="../data/novabank_customers.csv")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--default-rate", type=float, default=0.09)
    args = parser.parse_args()

    df = generate_customers(n=args.n, seed=args.seed, target_default_rate=args.default_rate)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} customers -> {args.out}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
