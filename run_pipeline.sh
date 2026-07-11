#!/usr/bin/env bash
# NovaBank AI Underwriting Copilot — full pipeline runner
# Regenerates the synthetic dataset, engineers features, trains models,
# and computes SHAP explanations. Run this before launching the Streamlit app.
set -e

cd "$(dirname "$0")/src"

echo "==> Step 1: Simulating 10,000 NovaBank customers..."
python3 simulator.py --n 10000 --out ../data/novabank_customers.csv --default-rate 0.09

echo ""
echo "==> Step 2: Engineering banking features..."
python3 feature_engineering.py --in ../data/novabank_customers.csv --out ../data/novabank_features.csv

echo ""
echo "==> Step 3: Training Logistic Regression / Random Forest / XGBoost..."
python3 train_model.py --in ../data/novabank_features.csv --model-dir ../models

echo ""
echo "==> Step 4: Computing SHAP explanations..."
python3 explain.py --model ../models/best_model.pkl --data ../models/test_split.csv --out-dir ../reports

echo ""
echo "==> Step 5: Running fairness & bias analysis..."
python3 fairness_check.py --model ../models/best_model.pkl --data ../models/test_split.csv --out-dir ../reports

echo ""
echo "==> Pipeline complete. Launch the app with:"
echo "    export ANTHROPIC_API_KEY=your_key_here   # optional, for LLM-generated reports"
echo "    streamlit run app/app.py"
