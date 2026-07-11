"""
NovaBank AI Underwriting Copilot
===================================
Takes a customer profile, model PD, and SHAP explanation, and produces a
professional underwriting report a loan officer can act on:
  - Risk level (Low / Medium / High)
  - Recommendation (Approve / Refer for Manual Review / Decline)
  - Key reasons (in plain business language, grounded in the SHAP values)
  - Suggested improvements for the applicant

Supports three interchangeable backends, tried in this order:
  1. Anthropic API (Claude) — cloud, needs ANTHROPIC_API_KEY
  2. Ollama (local open-source model, e.g. llama3.1, mistral, gemma2) — free,
     runs entirely on your machine, needs the Ollama app running locally
  3. Rule-based generator — deterministic, no network/model required at all

Usage (standalone test):
    python copilot.py
"""

import json
import os
import textwrap

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

CLAUDE_MODEL = "claude-sonnet-4-6"

# Ollama runs a local HTTP server (installed separately: https://ollama.com).
# Pull a model once with e.g. `ollama pull llama3.1`, then it's available here
# with no API key at all.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


def _ollama_is_running(base_url: str = OLLAMA_BASE_URL, timeout: float = 1.0) -> bool:
    if not _REQUESTS_AVAILABLE:
        return False
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _risk_bucket(pd_value: float) -> str:
    if pd_value < 0.08:
        return "Low"
    elif pd_value < 0.22:
        return "Medium"
    else:
        return "High"


def _rule_based_recommendation(pd_value: float) -> str:
    if pd_value < 0.08:
        return "Approve"
    elif pd_value < 0.22:
        return "Refer for Manual Review"
    else:
        return "Decline"


def build_prompt(customer: dict, pd_value: float, top_shap_features: list, risk_level: str, recommendation: str) -> str:
    """Builds the LLM prompt from customer profile + PD + SHAP explanation."""

    readable_shap = "\n".join(
        f"  - {f['feature'].replace('num__', '').replace('cat__', '').replace('_', ' ').title()}: "
        f"{'increases' if f['shap_value'] > 0 else 'decreases'} risk "
        f"(impact score {f['shap_value']:+.3f})"
        for f in top_shap_features
    )

    profile_lines = "\n".join(f"  - {k.replace('_', ' ').title()}: {v}" for k, v in customer.items())

    prompt = f"""
You are an underwriting analyst copilot for NovaBank, a retail lender in India.
You assist (never replace) human loan officers by turning a model's default-risk
prediction and its explanation into a clear, professional underwriting report.

CUSTOMER PROFILE:
{profile_lines}

MODEL OUTPUT:
  - Predicted Probability of Default (PD): {pd_value:.1%}
  - Risk Level (already determined by NovaBank's policy thresholds): {risk_level}
  - Recommendation (already determined by NovaBank's policy thresholds): {recommendation}

TOP RISK DRIVERS (from SHAP explainability, ordered by impact magnitude):
{readable_shap}

Write an underwriting report with these exact sections:
1. **Risk Level** — state exactly "{risk_level}" (this is fixed by policy — do not change it), with a
   one-line justification grounded in the PD and SHAP drivers above.
2. **Recommendation** — state exactly "{recommendation}" (this is fixed by policy — do not change it),
   with a one-line rationale.
3. **Key Reasons** — 3-5 bullet points in plain business language explaining what is driving this PD,
   grounded strictly in the SHAP drivers above. Do not invent factors not present in the data.
4. **Suggested Improvements** — 2-4 concrete, actionable suggestions the applicant could take to
   improve their risk profile for a future application (e.g. reduce credit utilization, add a
   co-applicant, build a longer credit history). Skip this section if risk is already Low.

Keep the tone professional, concise, and free of overclaiming. This is a decision-support
tool, not a final decision — end the report with a one-line disclaimer that final approval
rests with the human loan officer.
""".strip()
    return prompt


def generate_report_llm(customer: dict, pd_value: float, top_shap_features: list, api_key: str = None) -> str:
    """Calls the Anthropic API to generate the underwriting report."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No ANTHROPIC_API_KEY found in environment or arguments.")

    client = anthropic.Anthropic(api_key=key)
    risk_level = _risk_bucket(pd_value)
    recommendation = _rule_based_recommendation(pd_value)
    prompt = build_prompt(customer, pd_value, top_shap_features, risk_level, recommendation)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def generate_report_ollama(
    customer: dict, pd_value: float, top_shap_features: list,
    base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL,
) -> str:
    """Calls a local open-source model served by Ollama (no API key needed)."""
    if not _REQUESTS_AVAILABLE:
        raise RuntimeError("The 'requests' package is required for the Ollama backend (pip install requests).")

    risk_level = _risk_bucket(pd_value)
    recommendation = _rule_based_recommendation(pd_value)
    prompt = build_prompt(customer, pd_value, top_shap_features, risk_level, recommendation)

    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def generate_report_rule_based(customer: dict, pd_value: float, top_shap_features: list) -> str:
    """Deterministic fallback report generator — no API key / network required."""
    risk = _risk_bucket(pd_value)
    rec = _rule_based_recommendation(pd_value)

    positive_drivers = [f for f in top_shap_features if f["shap_value"] > 0][:5]
    negative_drivers = [f for f in top_shap_features if f["shap_value"] < 0][:5]

    def fmt(f):
        name = f["feature"].replace("num__", "").replace("cat__", "").replace("_", " ").title()
        return f"- {name} (impact {f['shap_value']:+.3f})"

    reasons = []
    if positive_drivers:
        reasons.append("**Factors increasing risk:**")
        reasons.extend(fmt(f) for f in positive_drivers)
    if negative_drivers:
        reasons.append("**Factors reducing risk:**")
        reasons.extend(fmt(f) for f in negative_drivers)
    reasons_block = "\n".join(reasons) if reasons else "- No dominant single factor identified."

    suggestions = []
    if risk != "Low":
        if any("credit_utilization" in f["feature"] or "emi_to_income" in f["feature"] for f in positive_drivers):
            suggestions.append("- Reduce outstanding credit card balances to lower utilization before reapplying.")
        if any("income_volatility" in f["feature"] for f in positive_drivers):
            suggestions.append("- Provide additional income-stability documentation (e.g. longer employment history, additional income proof).")
        if any("credit_history" in f["feature"] or "credit_maturity" in f["feature"] for f in positive_drivers):
            suggestions.append("- Build a longer credit history through consistent, on-time repayments.")
        if not suggestions:
            suggestions.append("- Consider a smaller loan amount or a co-applicant to strengthen the application.")

    suggestions_block = "\n".join(suggestions) if suggestions else ""

    report = f"""
### NovaBank Underwriting Report (rule-based fallback — no LLM API key configured)

**1. Risk Level:** {risk} (Predicted PD: {pd_value:.1%})

**2. Recommendation:** {rec}

**3. Key Reasons:**
{reasons_block}

{"**4. Suggested Improvements:**" + chr(10) + suggestions_block if suggestions_block else ""}

*This is an automated decision-support output. Final approval rests with the human loan officer.*
""".strip()
    return report


def generate_underwriting_report(
    customer: dict, pd_value: float, top_shap_features: list,
    api_key: str = None, engine: str = "auto",
) -> dict:
    """
    Main entry point. `engine` controls which backend generates the report:
      - "auto"       : Claude if an API key is set, else Ollama if it's running
                       locally, else the rule-based fallback (default)
      - "anthropic"  : force Claude (requires api_key or ANTHROPIC_API_KEY)
      - "ollama"     : force the local open-source model via Ollama
      - "rule_based" : force the deterministic, no-model fallback
    Returns a dict with the report text and metadata (including which
    engine actually produced it, in "engine_used").
    """
    risk_level = _risk_bucket(pd_value)
    recommendation = _rule_based_recommendation(pd_value)
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    engine_used = "rule_based"
    report_text = None

    def try_anthropic():
        nonlocal report_text, engine_used
        report_text = generate_report_llm(customer, pd_value, top_shap_features, api_key=key)
        engine_used = "anthropic"

    def try_ollama():
        nonlocal report_text, engine_used
        report_text = generate_report_ollama(customer, pd_value, top_shap_features)
        engine_used = "ollama"

    try:
        if engine == "anthropic":
            try_anthropic()
        elif engine == "ollama":
            try_ollama()
        elif engine == "rule_based":
            pass  # falls through to rule-based below
        else:  # "auto"
            if _ANTHROPIC_AVAILABLE and key:
                try_anthropic()
            elif _ollama_is_running():
                try_ollama()
    except Exception as e:
        report_text = generate_report_rule_based(customer, pd_value, top_shap_features)
        report_text += f"\n\n_(⚠️ {engine_used or 'LLM'} call failed, used rule-based fallback: {e})_"
        engine_used = "rule_based"

    if report_text is None:
        report_text = generate_report_rule_based(customer, pd_value, top_shap_features)
        engine_used = "rule_based"

    return {
        "risk_level": risk_level,
        "recommendation": recommendation,
        "pd": pd_value,
        "report_text": report_text,
        "used_llm": engine_used in ("anthropic", "ollama"),
        "engine_used": engine_used,
    }


if __name__ == "__main__":
    # Standalone smoke test with a synthetic example
    sample_customer = {
        "age": 29,
        "occupation": "Salaried - Private",
        "monthly_salary": 42000,
        "credit_score": 610,
        "requested_loan_amount": 800000,
        "loan_purpose": "Personal",
    }
    sample_shap = [
        {"feature": "num__credit_score", "shap_value": 0.41},
        {"feature": "num__income_volatility", "shap_value": 0.28},
        {"feature": "num__emi_to_income_ratio", "shap_value": 0.19},
        {"feature": "num__bank_relationship_years", "shap_value": -0.15},
        {"feature": "num__credit_utilization_ratio", "shap_value": 0.12},
    ]
    result = generate_underwriting_report(sample_customer, 0.27, sample_shap)
    print(json.dumps({k: v for k, v in result.items() if k != "report_text"}, indent=2))
    print()
    print(result["report_text"])
