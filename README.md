# 🏦 NovaBank AI Underwriting Copilot

An explainable AI loan-underwriting system built on a **domain-driven synthetic banking
customer simulator** — because real banking data is confidential, so this project builds
its own realistic dataset from business rules instead of using a generic Kaggle CSV.

> "I designed a domain-driven banking customer simulator that generates realistic retail
> lending data using business rules. The generated dataset powers an explainable AI
> underwriting system integrating XGBoost, SHAP, and LLM-based underwriting reports."

## Pipeline

```
NovaBank Customer Simulator
        │
        ▼
Synthetic Banking Dataset (10,000 customers, 39 raw features)
        │
        ▼
Feature Engineering (loan-to-income, EMI-to-income, utilization, debt burden, stability...)
        │
        ▼
Loan Default Prediction Model (Logistic Regression / Random Forest / XGBoost)
        │
        ▼
SHAP Explainability (global + per-customer)
        │
        ▼
AI Underwriting Copilot (Claude-generated officer report, with rule-based fallback)
```

## Project Structure

```
novabank/
├── src/
│   ├── simulator.py            # Stage 1: customer simulator
│   ├── feature_engineering.py  # Stage 2: derived banking features
│   ├── train_model.py          # Stage 3: trains & compares 3 models, saves best
│   ├── explain.py              # Stage 4: SHAP global + per-customer explanations
│   ├── fairness_check.py       # Stage 4b: approval-rate / error-rate fairness audit by demographic group
│   └── copilot.py              # Stage 5: LLM underwriting report generator
├── app/
│   └── app.py                  # Streamlit officer dashboard (final product)
├── data/
│   ├── novabank_customers.csv  # raw simulated dataset
│   └── novabank_features.csv   # engineered features
├── models/
│   ├── best_model.pkl          # best-performing trained pipeline
│   ├── all_models.pkl          # all 3 trained pipelines
│   ├── metadata.json           # metrics, feature lists, split info
│   ├── train_split.csv / test_split.csv
├── reports/
│   ├── shap_summary_bar.png       # global feature importance
│   ├── shap_summary_beeswarm.png  # direction + magnitude of impact
│   └── global_feature_importance.json
├── run_pipeline.sh             # one-command: simulate -> features -> train -> explain
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the full pipeline

```bash
./run_pipeline.sh
```

This regenerates the dataset, engineers features, trains all three models, picks the
best one by ROC-AUC, and computes SHAP explanations. Takes under a minute.

Each stage can also be run individually:

```bash
cd src
python simulator.py --n 10000 --out ../data/novabank_customers.csv --default-rate 0.09
python feature_engineering.py --in ../data/novabank_customers.csv --out ../data/novabank_features.csv
python train_model.py --in ../data/novabank_features.csv --model-dir ../models
python explain.py --model ../models/best_model.pkl --data ../models/test_split.csv --out-dir ../reports
```

## Launching the app

```bash
streamlit run app/app.py
```

- Pick a sample applicant, or fill in the form manually.
- Click **Assess Application** to get: Probability of Default, Risk Level, a SHAP
  waterfall chart, and an AI-generated underwriting report.
- Optionally paste an **Anthropic API key** in the sidebar to get Claude-generated
  reports instead of the rule-based fallback (works fully offline either way).

To enable Claude-generated reports from the command line instead of pasting the key
in the UI:

```bash
export ANTHROPIC_API_KEY=your_key_here
streamlit run app/app.py
```

## Stage 1 — Customer Simulator

Instead of independently randomizing each column (which produces unrealistic data with
no internal correlation), the simulator builds each customer through a **dependency
chain**, so downstream fields are realistic functions of upstream ones:

```
Demographics -> Career -> Finance -> Credit History -> Loan Application -> Risk Score -> Default
```

For example: education level affects income potential → income and city tier affect
savings accumulation → savings and income stability affect credit score → credit score
and EMI burden affect probability of default. The final default label is drawn from a
logistic risk model calibrated to hit a target default rate (~9%), so the dataset has
realistic, learnable signal without being leaky or synthetic-looking.

**Output:** 10,000 customers × 39 columns, ~9% default rate.

## Stage 2 — Feature Engineering

Derives underwriting-relevant ratios from raw fields, e.g.:

| Feature | Meaning |
|---|---|
| `loan_to_income_ratio` | Requested loan ÷ annual income |
| `emi_to_income_ratio` | Total monthly EMI burden ÷ monthly income |
| `credit_utilization_ratio` | Credit card outstanding ÷ credit limit |
| `debt_burden_score` | Weighted composite of EMI ratio, utilization, loan count |
| `income_stability_score` | Composite of income volatility, experience, banking tenure |

## Stage 3 — Model Training

Trains and compares **Logistic Regression**, **Random Forest**, and **XGBoost** on an
80/20 stratified split, and automatically saves whichever scores highest on ROC-AUC.
Typical results on the generated dataset:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | ~0.92 | ~0.64 | ~0.34 | ~0.86 | ~0.49 |
| Random Forest | ~0.91 | ~0.61 | ~0.42 | ~0.74 | ~0.54 |
| XGBoost | ~0.91 | ~0.61 | ~0.41 | ~0.72 | ~0.53 |

(Class weighting / `scale_pos_weight` is used throughout since defaults are a ~9%
minority class.)

## Stage 4 — SHAP Explainability

Automatically picks the right SHAP explainer for whichever model won (`LinearExplainer`
for Logistic Regression, `TreeExplainer` for Random Forest/XGBoost), and produces:
- A **global bar chart** of mean |SHAP value| per feature (which features matter most overall)
- A **beeswarm plot** showing direction and magnitude of each feature's impact
- **Per-customer waterfall explanations** in the app, showing exactly why one applicant's
  PD is high or low

## Stage 4b — Fairness & Bias Analysis

Loan-underwriting models carry real regulatory and ethical weight, so the pipeline includes a
dedicated fairness audit (`src/fairness_check.py`) that checks whether the model treats
demographic groups consistently on the held-out test set:

- **Approval / decline rates** per group (using the same PD thresholds as the app's policy)
- **False positive rate** (good payers wrongly flagged high-risk) and **recall** (actual
  defaulters correctly caught) per group — an "equalized odds" style check
- The **four-fifths rule**: a common fair-lending screening heuristic (if the lowest-approval
  group's rate is under 80% of the highest-approval group's, it's flagged for review)

Results are surfaced in the Streamlit app's "Global Model Insights" tab, alongside the raw
CSVs and bar charts saved to `reports/`. This is a screening signal, not a legal or definitive
verdict — a flagged group may reflect legitimate underlying risk differences, and any real flag
should be investigated further, not automatically acted on.

## Stage 5 — AI Underwriting Copilot

Feeds the customer profile, predicted PD, and top SHAP drivers into Claude, which returns
a structured report: **Risk Level**, **Recommendation**, **Key Reasons** (grounded in the
actual SHAP values — the prompt explicitly forbids inventing factors), and **Suggested
Improvements**. If no API key is configured, a deterministic rule-based generator produces
an equivalent (if less fluent) report, so the whole system works offline.

## Tech Stack

Python · NumPy · Pandas · Scikit-learn · XGBoost · SHAP · Streamlit · Anthropic (Claude) API · Faker

## Notes & Limitations

- All data is **synthetic** — generated from business-rule simulations, not real
  banking records. It's designed to have realistic-looking distributions and
  correlations for portfolio/demo purposes, not for production credit decisioning.
- The AI Underwriting Copilot is a **decision-support tool**. Every report ends with an
  explicit disclaimer that final approval rests with a human loan officer.
