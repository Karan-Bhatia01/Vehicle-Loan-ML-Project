"""Streamlit app for vehicle loan default risk scoring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.config import PROJECT_ROOT
from src.inference.predict import load_pipeline, predict

# --- Page config ---
st.set_page_config(
    page_title="Loan Risk Scorer",
    page_icon="🏦",
    layout="centered",
)

# --- Custom styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    max-width: 720px;
    padding-top: 2rem;
}

h1 {
    font-weight: 700;
    font-size: 1.8rem;
    letter-spacing: -0.02em;
    color: #0F1923;
    margin-bottom: 0.25rem;
}

.subtitle {
    color: #6B7280;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    line-height: 1.5;
}

.verdict-card {
    padding: 1.5rem 2rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    text-align: center;
}

.verdict-approve {
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
}
.verdict-approve h2 { color: #065F46; margin: 0; font-size: 1.4rem; }
.verdict-approve p { color: #047857; margin: 0.25rem 0 0 0; }

.verdict-review {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
}
.verdict-review h2 { color: #92400E; margin: 0; font-size: 1.4rem; }
.verdict-review p { color: #B45309; margin: 0.25rem 0 0 0; }

.verdict-reject {
    background: #FEF2F2;
    border: 1px solid #FECACA;
}
.verdict-reject h2 { color: #991B1B; margin: 0; font-size: 1.4rem; }
.verdict-reject p { color: #DC2626; margin: 0.25rem 0 0 0; }

.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.metric-box .label {
    font-size: 0.75rem;
    color: #6B7280;
    text-transform: none;
    letter-spacing: 0;
}
.metric-box .value {
    font-size: 1.25rem;
    font-weight: 600;
    color: #0F1923;
    margin-top: 0.15rem;
}
</style>
""", unsafe_allow_html=True)


# --- Load model ---
@st.cache_resource
def get_pipeline():
    path = PROJECT_ROOT / "models" / "loan_default_pipeline.pkl"
    if not path.exists():
        return None
    return load_pipeline(path)


pipeline = get_pipeline()

# --- Header ---
st.markdown("# 🏦 Loan Risk Scorer")
st.markdown('<p class="subtitle">Enter loan application details below. The model returns a calibrated default probability and a recommendation.</p>', unsafe_allow_html=True)

if pipeline is None:
    st.error("No model found. Run `python pipelines/train_pipeline.py` first to generate the pickle.")
    st.stop()

# --- Input form ---
with st.form("loan_form"):
    st.markdown("##### Loan details")
    col1, col2, col3 = st.columns(3)
    with col1:
        disbursed_amount = st.number_input("Disbursed amount (₹)", min_value=0, value=500000, step=10000)
        ltv = st.number_input("LTV ratio", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    with col2:
        asset_cost = st.number_input("Asset cost (₹)", min_value=0, value=650000, step=10000)
        age = st.number_input("Age at disbursal", min_value=18, max_value=75, value=35)
    with col3:
        bureau_score = st.number_input("Bureau score", min_value=0, max_value=900, value=650)
        credit_history = st.number_input("Credit history (months)", min_value=0, value=36)

    st.markdown("##### Bureau history")
    col4, col5, col6 = st.columns(3)
    with col4:
        total_accounts = st.number_input("Total accounts", min_value=0, value=5)
        active_accounts = st.number_input("Active accounts", min_value=0, value=3)
    with col5:
        overdue_accounts = st.number_input("Overdue accounts", min_value=0, value=0)
        inquiry_count = st.number_input("Inquiries", min_value=0, value=2)
    with col6:
        current_balance = st.number_input("Current balance (₹)", min_value=0, value=200000, step=10000)
        sanctioned_amount = st.number_input("Sanctioned amount (₹)", min_value=0, value=400000, step=10000)

    st.markdown("##### Derived risk fields")
    col7, col8, col9 = st.columns(3)
    with col7:
        installment_amount = st.number_input("Total installment (₹)", min_value=0, value=15000, step=500)
        overdue_ratio = st.number_input("Overdue ratio", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    with col8:
        credit_utilization = st.number_input("Credit utilization", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
        new_accounts_6m = st.number_input("New accounts (6m)", min_value=0, value=1)
    with col9:
        branch_id = st.number_input("Branch ID", min_value=0, value=100)
        manufacturer_id = st.number_input("Manufacturer ID", min_value=0, value=50)

    submitted = st.form_submit_button("Score this application", use_container_width=True)

# --- Prediction ---
if submitted:
    row = pd.DataFrame([{
        "branch_id": branch_id,
        "supplier_id": 0,
        "manufacturer_id": manufacturer_id,
        "pincode_id": 0,
        "state_id": 0,
        "employee_id": 0,
        "disbursed_amount": disbursed_amount,
        "asset_cost": asset_cost,
        "ltv": ltv,
        "bureau_score": bureau_score,
        "age_at_disbursal": age,
        "credit_history_months": credit_history,
        "total_account_count": total_accounts,
        "total_active_accounts": active_accounts,
        "total_overdue_accounts": overdue_accounts,
        "total_current_balance": current_balance,
        "total_sanctioned_amount": sanctioned_amount,
        "total_installment_amount": installment_amount,
        "inquiry_count": inquiry_count,
        "overdue_ratio": overdue_ratio,
        "credit_utilization": credit_utilization,
        "new_accounts_6m": new_accounts_6m,
    }])

    result = predict(pipeline, row)
    prob = result["probability"][0]
    decision = result["decision"][0]
    threshold = result["threshold"]

    # Anything within 30% below the reject threshold goes to manual review.
    if prob < threshold * 0.7:
        tier, css_class, label = "Low Risk", "verdict-approve", "Approve"
    elif prob < threshold:
        tier, css_class, label = "Medium Risk", "verdict-review", "Manual Review"
    else:
        tier, css_class, label = "High Risk", "verdict-reject", "Reject"

    st.markdown(f"""
    <div class="verdict-card {css_class}">
        <h2>{label}</h2>
        <p>{tier} — {prob:.1%} predicted default probability</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="label">Default probability</div>
            <div class="value">{prob:.2%}</div>
        </div>
        <div class="metric-box">
            <div class="label">Decision threshold</div>
            <div class="value">{threshold:.2%}</div>
        </div>
        <div class="metric-box">
            <div class="label">Model</div>
            <div class="value">{pipeline['model_name']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
