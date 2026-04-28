import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise – Loan Approval System",
    page_icon="💳",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1a237e 0%, #283593 60%, #3949ab 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .header-banner h1 { font-size: 2.4rem; margin: 0; font-weight: 800; }
    .header-banner p  { font-size: 1.05rem; margin-top: 0.4rem; opacity: 0.85; }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1a237e;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 1rem;
        border-left: 4px solid #3949ab;
        padding-left: 0.7rem;
    }

    /* Result boxes */
    .result-approved {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #43a047;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
    }
    .result-rejected {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border: 2px solid #e53935;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
    }
    .result-approved h2 { color: #2e7d32; font-size: 2rem; margin: 0; }
    .result-rejected h2 { color: #c62828; font-size: 2rem; margin: 0; }
    .result-approved p  { color: #388e3c; font-size: 1rem; }
    .result-rejected p  { color: #c62828; font-size: 1rem; }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; }
    .metric-card {
        flex: 1; min-width: 140px;
        background: #f3f4f6;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        text-align: center;
    }
    .metric-card .metric-val { font-size: 1.6rem; font-weight: 800; color: #1a237e; }
    .metric-card .metric-lbl { font-size: 0.75rem; color: #666; margin-top: 0.2rem; }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #3949ab);
        color: white; border: none;
        border-radius: 10px;
        padding: 0.75rem 2.5rem;
        font-size: 1.05rem; font-weight: 700;
        width: 100%; cursor: pointer;
        transition: transform 0.15s;
    }
    .stButton > button:hover { transform: translateY(-2px); }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #1a237e; }
    [data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("model.pkl")
    scaler        = joblib.load("scaler.pkl")
    ohe           = joblib.load("ohe.pkl")
    le_edu        = joblib.load("le_edu.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, ohe, le_edu, feature_names

model, scaler, ohe, le_edu, feature_names = load_artifacts()


# ── Sidebar – model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 CreditWise")
    st.markdown("---")
    st.markdown("**Model:** Naïve Bayes")
    st.markdown("**Accuracy:** 86.0 %")
    st.markdown("**Precision:** 81.1 %")
    st.markdown("**Recall:** 70.5 %")
    st.markdown("**F1 Score:** 75.4 %")
    st.markdown("---")
    st.markdown("**Features used**")
    st.markdown(
        "Income · Co-applicant income · Age · Dependents · "
        "Credit score · DTI ratio · Savings · Collateral · "
        "Loan amount & term · Employment · Marital status · "
        "Loan purpose · Property area · Education · Gender · "
        "Employer category"
    )
    st.markdown("---")
    st.caption("Built with Streamlit × scikit-learn")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>💳 CreditWise Loan Approval System</h1>
    <p>Fill in the applicant details below and click <strong>Predict</strong> to get an instant loan decision.</p>
</div>
""", unsafe_allow_html=True)


# ── Example pre-fill via session state ────────────────────────────────────────
if "example_loaded" not in st.session_state:
    st.session_state.example_loaded = False

col_ex1, col_ex2 = st.columns([1, 4])
with col_ex1:
    if st.button("✨ Load Approved Example"):
        st.session_state.example_loaded = True
        st.session_state.ex = {
            "gender": "Male", "age": 35, "marital_status": "Married",
            "dependents": 1, "education": "Graduate", "employment": "Salaried",
            "applicant_income": 150000, "coapplicant_income": 50000,
            "credit_score": 800, "dti_ratio": 15.0,
            "savings": 500000, "collateral_value": 2000000,
            "loan_amount": 500000, "loan_term": 360,
            "loan_purpose": "Home", "property_area": "Urban",
            "employer_cat": "Government", "existing_loans": 0,
        }
        st.rerun()

ex = st.session_state.get("ex", {})

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("loan_form"):

    # Personal Information
    st.markdown('<div class="section-card"><div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        gender_opts = ["Male", "Female"]
        gender      = st.selectbox("Gender", gender_opts, index=gender_opts.index(ex.get("gender", "Male")))
        age         = st.number_input("Age", 18, 75, ex.get("age", 35))
    with c2:
        ms_opts        = ["Single", "Married"]
        marital_status = st.selectbox("Marital Status", ms_opts, index=ms_opts.index(ex.get("marital_status", "Single")))
        dependents     = st.number_input("Number of Dependents", 0, 10, ex.get("dependents", 0))
    with c3:
        edu_opts  = ["Graduate", "Not Graduate"]
        education = st.selectbox("Education Level", edu_opts, index=edu_opts.index(ex.get("education", "Graduate")))
        emp_opts  = ["Salaried", "Self-employed", "Contract", "Unemployed"]
        employment = st.selectbox("Employment Status", emp_opts, index=emp_opts.index(ex.get("employment", "Salaried")))
    st.markdown('</div>', unsafe_allow_html=True)

    # Financial Information
    st.markdown('<div class="section-card"><div class="section-title">💰 Financial Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        applicant_income   = st.number_input("Applicant Income (₹)", 0, 10_000_000, ex.get("applicant_income", 50_000), step=1000)
        coapplicant_income = st.number_input("Co-applicant Income (₹)", 0, 10_000_000, ex.get("coapplicant_income", 0), step=1000)
    with c2:
        credit_score = st.slider("Credit Score", 300, 900, ex.get("credit_score", 650))
        dti_ratio    = st.slider("DTI Ratio (%)", 0.0, 100.0, float(ex.get("dti_ratio", 30.0)), step=0.5)
    with c3:
        savings          = st.number_input("Savings (₹)", 0, 100_000_000, ex.get("savings", 100_000), step=10_000)
        collateral_value = st.number_input("Collateral Value (₹)", 0, 100_000_000, ex.get("collateral_value", 500_000), step=10_000)
    st.markdown('</div>', unsafe_allow_html=True)

    # Loan Details
    st.markdown('<div class="section-card"><div class="section-title">🏦 Loan Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amount = st.number_input("Loan Amount (₹)", 10_000, 100_000_000, ex.get("loan_amount", 500_000), step=10_000)
        term_opts   = [12, 24, 36, 48, 60, 84, 120, 180, 240, 360]
        loan_term   = st.selectbox("Loan Term (months)", term_opts, index=term_opts.index(ex.get("loan_term", 60)))
    with c2:
        pur_opts     = ["Home", "Car", "Education", "Business", "Personal"]
        loan_purpose = st.selectbox("Loan Purpose", pur_opts, index=pur_opts.index(ex.get("loan_purpose", "Home")))
        area_opts    = ["Urban", "Semiurban", "Rural"]
        property_area = st.selectbox("Property Area", area_opts, index=area_opts.index(ex.get("property_area", "Urban")))
    with c3:
        emp_cat_opts = ["Private", "Government", "MNC", "Business", "Unemployed"]
        employer_cat = st.selectbox("Employer Category", emp_cat_opts, index=emp_cat_opts.index(ex.get("employer_cat", "Private")))
        existing_loans = st.number_input("Existing Loans", 0, 20, ex.get("existing_loans", 0))
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍  Predict Loan Approval")


# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    # --- Build raw input dict ---
    input_dict = {
        "Applicant_Income":    applicant_income,
        "Coapplicant_Income":  coapplicant_income,
        "Age":                 age,
        "Dependents":          dependents,
        "Credit_Score":        credit_score,
        "Existing_Loans":      existing_loans,
        "DTI_Ratio":           dti_ratio,
        "Savings":             savings,
        "Collateral_Value":    collateral_value,
        "Loan_Amount":         loan_amount,
        "Loan_Term":           loan_term,
        "Education_Level":     education,
        "Employment_Status":   employment,
        "Marital_Status":      marital_status,
        "Loan_Purpose":        loan_purpose,
        "Property_Area":       property_area,
        "Gender":              gender,
        "Employer_Category":   employer_cat,
    }
    input_df = pd.DataFrame([input_dict])

    # --- Encode Education ---
    input_df["Education_Level"] = le_edu.transform(input_df["Education_Level"])

    # --- OHE ---
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]
    encoded    = ohe.transform(input_df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))
    input_df   = pd.concat([input_df.drop(columns=ohe_cols).reset_index(drop=True),
                             encoded_df.reset_index(drop=True)], axis=1)

    # --- Feature engineering ---
    input_df["DTI_Ratio_sq"]         = input_df["DTI_Ratio"] ** 2
    input_df["Credit_Score_sq"]      = input_df["Credit_Score"] ** 2
    input_df["Applicant_Income_log"] = np.log1p(input_df["Applicant_Income"])

    # Drop raw cols used only for engineering
    input_df = input_df.drop(columns=["Credit_Score", "DTI_Ratio"], errors="ignore")

    # Align columns
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # --- Scale & predict ---
    X_scaled   = scaler.transform(input_df)
    prediction = model.predict(X_scaled)[0]
    proba      = model.predict_proba(X_scaled)[0]

    approval_prob = proba[1] * 100
    reject_prob   = proba[0] * 100

    # --- Show result ---
    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    col_res, col_detail = st.columns([1, 1])

    with col_res:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-approved">
                <h2>✅ APPROVED</h2>
                <p>This loan application is likely to be <strong>approved</strong>.</p>
                <p style="font-size:1.5rem; font-weight:800; color:#2e7d32;">{approval_prob:.1f}% confidence</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-rejected">
                <h2>❌ REJECTED</h2>
                <p>This loan application is likely to be <strong>rejected</strong>.</p>
                <p style="font-size:1.5rem; font-weight:800; color:#c62828;">{reject_prob:.1f}% confidence</p>
            </div>""", unsafe_allow_html=True)

    with col_detail:
        st.markdown('<div class="section-card"><div class="section-title">📋 Application Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-val">₹{loan_amount:,.0f}</div>
                <div class="metric-lbl">Loan Amount</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{credit_score}</div>
                <div class="metric-lbl">Credit Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{dti_ratio:.1f}%</div>
                <div class="metric-lbl">DTI Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">₹{applicant_income:,.0f}</div>
                <div class="metric-lbl">Income</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Probability bar
        st.markdown("**Approval Probability**")
        st.progress(int(approval_prob))
        st.caption(f"Approve: {approval_prob:.1f}%  |  Reject: {reject_prob:.1f}%")

        # Tips
        if prediction == 0:
            st.markdown("**💡 Tips to improve chances:**")
            if credit_score < 650:
                st.info("📈 Improve your credit score above 700")
            if dti_ratio > 40:
                st.info("📉 Reduce your DTI ratio below 40%")
            if applicant_income < 30000:
                st.info("💼 Higher income or co-applicant income helps")
