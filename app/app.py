"""
NovaBank AI Underwriting Copilot — Streamlit App
===================================================
Run with:
    streamlit run app.py

The loan officer enters (or picks a sample) customer details, the app:
  1. Predicts Probability of Default using the trained model
  2. Explains the prediction using SHAP
  3. Generates a professional AI underwriting report (Claude, with a
     rule-based fallback if no ANTHROPIC_API_KEY is set)
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from copilot import generate_underwriting_report  # noqa: E402
from explain import compute_shap_values, top_features_for_row  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "models", "test_split.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "novabank_features.csv")

st.set_page_config(page_title="NovaBank AI Underwriting Copilot", layout="wide", page_icon="🏦")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_data
def load_sample_customers(n=25):
    df = pd.read_csv(TEST_SPLIT_PATH)
    return df.sample(n, random_state=1).reset_index(drop=True)


@st.cache_resource
def load_background_data(sample_size=300):
    df = pd.read_csv(TEST_SPLIT_PATH)
    if "default" in df.columns:
        df = df.drop(columns=["default"])
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=1)
    return df


def predict_pd(pipeline, customer_df):
    return float(pipeline.predict_proba(customer_df)[:, 1][0])


def explain_customer(pipeline, customer_df, background_df, top_n=8):
    preprocessor = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    background_transformed = preprocessor.transform(background_df)
    feature_names = list(preprocessor.get_feature_names_out())
    background_transformed_df = pd.DataFrame(background_transformed, columns=feature_names)

    customer_transformed = preprocessor.transform(customer_df)
    customer_transformed_df = pd.DataFrame(customer_transformed, columns=feature_names)

    clf_name = type(clf).__name__
    if clf_name in ("XGBClassifier", "RandomForestClassifier"):
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.LinearExplainer(clf, background_transformed_df)

    raw_shap = explainer.shap_values(customer_transformed_df)
    if isinstance(raw_shap, list):
        shap_row = raw_shap[1][0]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_row = raw_shap[0, :, 1]
    else:
        shap_row = raw_shap[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        base_value = float(np.array(expected_value).reshape(-1)[-1])
    else:
        base_value = float(expected_value)

    top_features = top_features_for_row(shap_row, feature_names, top_n=top_n)
    return top_features, shap_row, feature_names, customer_transformed_df, base_value


def render_shap_waterfall(shap_row, feature_names, base_value, customer_transformed_df):
    explanation = shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=customer_transformed_df.iloc[0].values,
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.tight_layout()
    return fig


def risk_color(risk_level):
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "⚪")


def engine_label(engine_used: str) -> str:
    return {
        "anthropic": "Claude (Anthropic API)",
        "ollama": "a local open-source model (Ollama)",
        "rule_based": "the rule-based fallback (no LLM)",
    }.get(engine_used, engine_used)


def main():
    st.title("🏦 NovaBank AI Underwriting Copilot")
    st.caption(
        "Explainable AI loan underwriting — synthetic data, trained ML model, SHAP explanations, "
        "and an LLM-generated officer report."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "No trained model found. Run the pipeline first:\n\n"
            "```\ncd src\npython simulator.py\npython feature_engineering.py\npython train_model.py\n```"
        )
        return

    pipeline = load_model()
    metadata = load_metadata()
    background_df = load_background_data()

    with st.sidebar:
        st.header("⚙️ Model Info")
        st.write(f"**Best model:** {metadata['best_model'].replace('_', ' ').title()}")
        st.write(f"**Test ROC-AUC:** {metadata['results'][metadata['best_model']]['roc_auc']}")
        st.write(f"**Test set default rate:** {metadata['default_rate_test']:.1%}")
        st.divider()

        engine_choice = st.selectbox(
            "AI report engine",
            ["Auto (Claude → Ollama → rule-based)", "Force Claude", "Force Ollama (local)", "Force rule-based"],
            help="Auto uses Claude if an API key is set, otherwise a local Ollama model if it's running, "
                 "otherwise a deterministic rule-based report.",
        )
        engine_map = {
            "Auto (Claude → Ollama → rule-based)": "auto",
            "Force Claude": "anthropic",
            "Force Ollama (local)": "ollama",
            "Force rule-based": "rule_based",
        }
        engine = engine_map[engine_choice]

        api_key = st.text_input(
            "Anthropic API key (optional — for LLM-generated reports)",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            help="If left blank, a rule-based fallback report is generated instead of calling Claude.",
        )
        st.divider()
        st.caption("Tip: pick a sample customer below, or fill the form manually.")

    tab1, tab2 = st.tabs(["📋 New Loan Application", "📊 Global Model Insights"])

    with tab1:
        sample_df = load_sample_customers()
        col_pick, col_reset = st.columns([4, 1])
        with col_pick:
            pick_idx = st.selectbox(
                "Load a sample applicant (optional)",
                options=["-- Enter manually --"] + [f"Sample #{i}" for i in range(len(sample_df))],
            )

        if pick_idx != "-- Enter manually --":
            row = sample_df.iloc[int(pick_idx.split("#")[1])]
        else:
            row = None

        st.subheader("Applicant Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 21, 65, int(row["age"]) if row is not None else 32)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                                   index=["Male", "Female", "Other"].index(row["gender"]) if row is not None else 0)
            city_tier = st.selectbox("City Tier", ["Tier-1", "Tier-2", "Tier-3"],
                                      index=["Tier-1", "Tier-2", "Tier-3"].index(row["city_tier"]) if row is not None else 0)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"],
                                           index=["Single", "Married", "Divorced", "Widowed"].index(row["marital_status"]) if row is not None else 0)
            dependents = st.number_input("Dependents", 0, 6, int(row["dependents"]) if row is not None else 0)
            education = st.selectbox("Education", ["High School", "Diploma", "Bachelor's", "Master's", "PhD/Professional"],
                                      index=["High School", "Diploma", "Bachelor's", "Master's", "PhD/Professional"].index(row["education"]) if row is not None else 2)
        with c2:
            occupation = st.selectbox(
                "Occupation",
                ["Salaried - Private", "Salaried - Government", "Salaried - PSU",
                 "Self-Employed - Professional", "Self-Employed - Business", "Gig/Freelance", "Unemployed/Student"],
                index=0 if row is None else ["Salaried - Private", "Salaried - Government", "Salaried - PSU",
                 "Self-Employed - Professional", "Self-Employed - Business", "Gig/Freelance", "Unemployed/Student"].index(row["occupation"]),
            )
            years_experience = st.number_input("Years of Experience", 0, 40, int(row["years_experience"]) if row is not None else 5)
            employer_type = st.selectbox("Employer Type", ["MNC", "Large Domestic Co.", "SME", "Startup", "Government", "Self-Employed", "Not Employed"],
                                          index=["MNC", "Large Domestic Co.", "SME", "Startup", "Government", "Self-Employed", "Not Employed"].index(row["employer_type"]) if row is not None else 0)
            job_switches = st.number_input("Job Switches", 0, 12, int(row["job_switches"]) if row is not None else 1)
            monthly_salary = st.number_input("Monthly Salary (₹)", 8000, 1200000, int(row["monthly_salary"]) if row is not None else 45000, step=1000)
            monthly_expenses = st.number_input("Monthly Expenses (₹)", 1000, 500000, int(row["monthly_expenses"]) if row is not None else 20000, step=500)
        with c3:
            credit_score = st.slider("Credit Score", 300, 900, int(row["credit_score"]) if row is not None else 680)
            requested_loan_amount = st.number_input("Requested Loan Amount (₹)", 20000, 15000000, int(row["requested_loan_amount"]) if row is not None else 500000, step=5000)
            loan_purpose = st.selectbox("Loan Purpose", ["Home", "Vehicle", "Personal", "Education", "Business Expansion", "Gold/Consumer Durable"],
                                         index=["Home", "Vehicle", "Personal", "Education", "Business Expansion", "Gold/Consumer Durable"].index(row["loan_purpose"]) if row is not None else 2)
            loan_tenure_months = st.number_input("Loan Tenure (months)", 6, 300, int(row["loan_tenure_months"]) if row is not None else 36)
            interest_rate_pct = st.number_input("Interest Rate (%)", 6.5, 24.0, float(row["interest_rate_pct"]) if row is not None else 12.0)
            collateral_available = st.checkbox("Collateral Available", value=bool(row["collateral_available"]) if row is not None else False)
            co_applicant = st.checkbox("Co-Applicant", value=bool(row["co_applicant"]) if row is not None else False)

        st.subheader("Credit History")
        c4, c5, c6 = st.columns(3)
        with c4:
            savings_balance = st.number_input("Savings Balance (₹)", 0, 20000000, int(row["savings_balance"]) if row is not None else 150000, step=5000)
            existing_investments = st.number_input("Existing Investments (₹)", 0, 10000000, int(row["existing_investments"]) if row is not None else 50000, step=5000)
            bank_relationship_years = st.number_input("Bank Relationship (years)", 0, 47, int(row["bank_relationship_years"]) if row is not None else 3)
            credit_history_length_years = st.number_input("Credit History Length (years)", 0, 47, int(row["credit_history_length_years"]) if row is not None else 3)
        with c5:
            num_credit_cards = st.number_input("Number of Credit Cards", 0, 8, int(row["num_credit_cards"]) if row is not None else 2)
            total_credit_limit = st.number_input("Total Credit Limit (₹)", 0, 2000000, int(row["total_credit_limit"]) if row is not None else 100000, step=5000)
            credit_card_outstanding = st.number_input("Credit Card Outstanding (₹)", 0, 2000000, int(row["credit_card_outstanding"]) if row is not None else 20000, step=1000)
            existing_loans_count = st.number_input("Existing Loans Count", 0, 5, int(row["existing_loans_count"]) if row is not None else 0)
        with c6:
            existing_emi = st.number_input("Existing EMI (₹)", 0, 200000, int(row["existing_emi"]) if row is not None else 0, step=500)
            past_defaults = st.number_input("Past Defaults", 0, 3, int(row["past_defaults"]) if row is not None else 0)
            late_payments_last_12m = st.number_input("Late Payments (last 12m)", 0, 12, int(row["late_payments_last_12m"]) if row is not None else 0)
            income_volatility = st.slider("Income Volatility", 0.0, 1.0, float(row["income_volatility"]) if row is not None else 0.20)

        if st.button("🔍 Assess Application", type="primary", use_container_width=True):
            emi_r = requested_loan_amount * (interest_rate_pct / 1200) * (1 + interest_rate_pct / 1200) ** loan_tenure_months / (((1 + interest_rate_pct / 1200) ** loan_tenure_months) - 1)

            customer_dict = {
                "age": age, "gender": gender, "city_tier": city_tier, "marital_status": marital_status,
                "dependents": dependents, "education": education, "occupation": occupation,
                "years_experience": years_experience, "employer_type": employer_type, "job_switches": job_switches,
                "monthly_salary": monthly_salary, "monthly_expenses": monthly_expenses,
                "disposable_income": monthly_salary - monthly_expenses,
                "income_volatility": income_volatility, "savings_balance": savings_balance,
                "existing_investments": existing_investments, "bank_relationship_years": bank_relationship_years,
                "credit_score": credit_score, "credit_history_length_years": credit_history_length_years,
                "num_credit_cards": num_credit_cards, "total_credit_limit": total_credit_limit,
                "credit_card_outstanding": credit_card_outstanding,
                "credit_utilization_ratio": (credit_card_outstanding / total_credit_limit) if total_credit_limit > 0 else 0,
                "existing_loans_count": existing_loans_count, "existing_emi": existing_emi,
                "past_defaults": past_defaults, "late_payments_last_12m": late_payments_last_12m,
                "loan_purpose": loan_purpose, "requested_loan_amount": requested_loan_amount,
                "loan_tenure_months": loan_tenure_months, "interest_rate_pct": interest_rate_pct,
                "emi": round(emi_r, 0), "collateral_available": collateral_available, "co_applicant": co_applicant,
                "loan_to_income_ratio": requested_loan_amount / (monthly_salary * 12),
                "emi_to_income_ratio": (emi_r + existing_emi) / monthly_salary,
                "total_debt_to_income_ratio": (emi_r + existing_emi + credit_card_outstanding * 0.03) / monthly_salary,
                "expense_to_income_ratio": monthly_expenses / monthly_salary,
                "disposable_income_after_loan": monthly_salary - monthly_expenses - emi_r - existing_emi,
                "savings_to_loan_ratio": (savings_balance / requested_loan_amount) if requested_loan_amount > 0 else 0,
                "debt_burden_score": 0.5 * min((emi_r + existing_emi) / monthly_salary, 2) + 0.3 * ((credit_card_outstanding / total_credit_limit) if total_credit_limit > 0 else 0) + 0.2 * min(existing_loans_count / 5, 1),
                "delinquency_score": past_defaults * 2 + late_payments_last_12m,
                "income_stability_score": (1 - income_volatility) * 0.6 + min(years_experience, 20) / 20 * 0.25 + min(bank_relationship_years, 20) / 20 * 0.15,
                "employment_tenure_ratio": years_experience / max(job_switches, 1),
                "credit_maturity_score": np.log1p(credit_history_length_years) / np.log1p(47),
                "loan_amount_to_salary_multiple": requested_loan_amount / monthly_salary,
                "has_collateral_flag": int(collateral_available), "has_co_applicant_flag": int(co_applicant),
            }

            feature_cols = metadata["feature_cols"]
            customer_df = pd.DataFrame([{k: customer_dict.get(k) for k in feature_cols}])

            pd_value = predict_pd(pipeline, customer_df)
            top_features, shap_row, feature_names, customer_transformed_df, base_val = explain_customer(
                pipeline, customer_df, background_df
            )

            st.divider()
            st.subheader("Assessment Result")

            m1, m2, m3 = st.columns(3)
            m1.metric("Probability of Default", f"{pd_value:.1%}")
            risk_lvl = "Low" if pd_value < 0.08 else ("Medium" if pd_value < 0.22 else "High")
            m2.metric("Risk Level", f"{risk_color(risk_lvl)} {risk_lvl}")
            rec = "Approve" if pd_value < 0.08 else ("Refer for Manual Review" if pd_value < 0.22 else "Decline")
            m3.metric("Suggested Decision", rec)

            colA, colB = st.columns([1, 1])
            with colA:
                st.markdown("**Top SHAP Drivers**")
                shap_display_df = pd.DataFrame(top_features).rename(
                    columns={"feature": "Feature", "shap_value": "SHAP Impact"}
                )
                shap_display_df["Feature"] = shap_display_df["Feature"].str.replace("num__", "").str.replace("cat__", "").str.replace("_", " ").str.title()
                st.dataframe(shap_display_df, use_container_width=True, hide_index=True)

            with colB:
                try:
                    fig = render_shap_waterfall(shap_row, feature_names, base_val, customer_transformed_df)
                    st.pyplot(fig)
                except Exception as e:
                    st.info(f"Waterfall plot unavailable for this model type. ({e})")

            st.markdown("### 🤖 AI Underwriting Report")
            with st.spinner("Generating report..."):
                result = generate_underwriting_report(
                    customer_dict, pd_value, top_features, api_key=api_key or None, engine=engine
                )
            if result["used_llm"]:
                st.success(f"Report generated by {engine_label(result['engine_used'])}.")
            else:
                st.info(
                    "Rule-based report (no LLM available/selected) — choose Claude or Ollama in the "
                    "sidebar for an LLM-generated report."
                )
            st.markdown(result["report_text"])

    with tab2:
        st.subheader("Global Model Insights")
        st.write("These explain what drives risk **across the whole customer base**, not just one applicant.")

        c1, c2 = st.columns(2)
        with c1:
            bar_path = os.path.join(BASE_DIR, "reports", "shap_summary_bar.png")
            if os.path.exists(bar_path):
                st.image(bar_path, caption="Global Feature Importance (mean |SHAP value|)")
        with c2:
            beeswarm_path = os.path.join(BASE_DIR, "reports", "shap_summary_beeswarm.png")
            if os.path.exists(beeswarm_path):
                st.image(beeswarm_path, caption="SHAP Beeswarm — direction & magnitude of impact")

        st.subheader("Model Comparison")
        results_df = pd.DataFrame(metadata["results"]).T
        st.dataframe(results_df, use_container_width=True)

        st.divider()
        st.subheader("⚖️ Fairness & Bias Analysis")
        st.write(
            "Checks whether the model approves/declines applicants at meaningfully different rates "
            "across demographic groups on the held-out test set. Uses the four-fifths rule (a common "
            "fair-lending screening heuristic): if the lowest-approval group's rate falls below 80% of "
            "the highest-approval group's rate, it's flagged for review — not an automatic finding of bias."
        )

        fairness_summary_path = os.path.join(BASE_DIR, "reports", "fairness_summary.json")
        if os.path.exists(fairness_summary_path):
            with open(fairness_summary_path) as f:
                fairness_summary = json.load(f)

            for group_col, result in fairness_summary.items():
                st.markdown(f"**By {group_col.replace('_', ' ').title()}**")
                fc1, fc2 = st.columns([1, 1])
                with fc1:
                    metrics_df = pd.DataFrame(result["metrics"])
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                with fc2:
                    plot_path = os.path.join(BASE_DIR, "reports", f"fairness_approval_rate_by_{group_col}.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path)

                rule = result["four_fifths_rule"]
                if rule.get("applicable"):
                    if rule["passes_four_fifths_rule"]:
                        st.success(
                            f"✅ Passes the four-fifths rule (impact ratio {rule['impact_ratio']:.2f}). "
                            f"Lowest approval rate: {rule['lowest_approval_group']} ({rule['lowest_approval_rate']:.1%}), "
                            f"highest: {rule['highest_approval_group']} ({rule['highest_approval_rate']:.1%})."
                        )
                    else:
                        st.warning(
                            f"⚠️ Fails the four-fifths rule (impact ratio {rule['impact_ratio']:.2f}) — "
                            f"worth a closer look. Lowest approval rate: {rule['lowest_approval_group']} "
                            f"({rule['lowest_approval_rate']:.1%}), highest: {rule['highest_approval_group']} "
                            f"({rule['highest_approval_rate']:.1%})."
                        )
        else:
            st.info(
                "No fairness report found. Run `python src/fairness_check.py` to generate one."
            )


if __name__ == "__main__":
    main()
