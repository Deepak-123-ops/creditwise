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
    c1, c2 = st.columns(2)
    with c1:
        gender_opts = ["Male", "Female"]
        gender      = st.selectbox("Gender", gender_opts, index=gender_opts.index(ex.get("gender", "Male")))
        st.caption("⚥ Used for demographic profiling")

        age = st.number_input("Age", 18, 75, ex.get("age", 35))
        if age < 25:
            st.caption("🟡 Young — limited credit history (18–24)")
        elif age < 40:
            st.caption("🟢 Prime — strong earning years (25–39)")
        elif age < 55:
            st.caption("🟢 Experienced — stable profile (40–54)")
        else:
            st.caption("🟡 Senior — shorter repayment horizon (55+)")

        ms_opts        = ["Single", "Married"]
        marital_status = st.selectbox("Marital Status", ms_opts, index=ms_opts.index(ex.get("marital_status", "Single")))
        if marital_status == "Married":
            st.caption("🟢 Married — co-applicant income likely")
        else:
            st.caption("🟡 Single — sole income dependency")

    with c2:
        dependents = st.number_input("Number of Dependents", 0, 10, ex.get("dependents", 0))
        if dependents == 0:
            st.caption("🟢 None — no financial dependents")
        elif dependents <= 2:
            st.caption("🟡 Low — manageable responsibility (1–2)")
        else:
            st.caption("🔴 High — more dependents = higher expenses (3+)")

        edu_opts  = ["Graduate", "Not Graduate"]
        education = st.selectbox("Education Level", edu_opts, index=edu_opts.index(ex.get("education", "Graduate")))
        if education == "Graduate":
            st.caption("🟢 Graduate — higher earning potential")
        else:
            st.caption("🟡 Not Graduate — may affect loan scoring")

        emp_opts   = ["Salaried", "Self-employed", "Contract", "Unemployed"]
        employment = st.selectbox("Employment Status", emp_opts, index=emp_opts.index(ex.get("employment", "Salaried")))
        if employment == "Salaried":
            st.caption("🟢 Salaried — most stable income type")
        elif employment == "Self-employed":
            st.caption("🟡 Self-employed — variable income")
        elif employment == "Contract":
            st.caption("🟠 Contract — less job security")
        else:
            st.caption("🔴 Unemployed — high risk for lenders")
    st.markdown('</div>', unsafe_allow_html=True)

    # Financial Information
    st.markdown('<div class="section-card"><div class="section-title">💰 Financial Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        applicant_income = st.number_input("Applicant Income (₹)", 0, 10_000_000, ex.get("applicant_income", 50_000), step=1)
        if applicant_income < 25_000:
            st.caption("🔴 Low — below ₹25,000")
        elif applicant_income < 75_000:
            st.caption("🟡 Moderate — ₹25,000 – ₹75,000")
        elif applicant_income < 2_00_000:
            st.caption("🟢 Good — ₹75,000 – ₹2,00,000")
        else:
            st.caption("🌟 Excellent — above ₹2,00,000")

        coapplicant_income = st.number_input("Co-applicant Income (₹)", 0, 10_000_000, ex.get("coapplicant_income", 0), step=1)
        if coapplicant_income == 0:
            st.caption("⚪ None — no co-applicant")
        elif coapplicant_income < 25_000:
            st.caption("🟡 Low — below ₹25,000")
        else:
            st.caption("🟢 Good — boosts approval chances")

        credit_score = st.slider("Credit Score", 300, 900, ex.get("credit_score", 650))
        if credit_score < 500:
            st.caption("🔴 Poor — very high risk (300–499)")
        elif credit_score < 600:
            st.caption("🟠 Fair — high risk (500–599)")
        elif credit_score < 700:
            st.caption("🟡 Average — moderate risk (600–699)")
        elif credit_score < 800:
            st.caption("🟢 Good — low risk (700–799)")
        else:
            st.caption("🌟 Excellent — very low risk (800–900)")

    with c2:
        dti_ratio = st.slider("DTI Ratio (%)", 0.0, 100.0, float(ex.get("dti_ratio", 30.0)), step=0.5)
        if dti_ratio < 20:
            st.caption("🌟 Excellent — very low debt burden (<20%)")
        elif dti_ratio < 35:
            st.caption("🟢 Good — manageable debt (20–35%)")
        elif dti_ratio < 50:
            st.caption("🟡 Moderate — borderline (35–50%)")
        else:
            st.caption("🔴 High — too much debt (>50%)")

        savings = st.number_input("Savings (₹)", 0, 100_000_000, ex.get("savings", 100_000), step=1)
        if savings < 10_000:
            st.caption("🔴 Very Low — below ₹10,000")
        elif savings < 1_00_000:
            st.caption("🟡 Low — ₹10,000 – ₹1,00,000")
        elif savings < 5_00_000:
            st.caption("🟢 Good — ₹1,00,000 – ₹5,00,000")
        else:
            st.caption("🌟 Excellent — above ₹5,00,000")

        collateral_value = st.number_input("Collateral Value (₹)", 0, 100_000_000, ex.get("collateral_value", 500_000), step=1)
        if collateral_value < 1_00_000:
            st.caption("🔴 Very Low — below ₹1,00,000")
        elif collateral_value < 10_00_000:
            st.caption("🟡 Moderate — ₹1,00,000 – ₹10,00,000")
        elif collateral_value < 50_00_000:
            st.caption("🟢 Good — ₹10,00,000 – ₹50,00,000")
        else:
            st.caption("🌟 Excellent — above ₹50,00,000")
    st.markdown('</div>', unsafe_allow_html=True)

    # Loan Details
    st.markdown('<div class="section-card"><div class="section-title">🏦 Loan Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        loan_amount = st.number_input("Loan Amount (₹)", 10_000, 100_000_000, ex.get("loan_amount", 500_000), step=1)
        if loan_amount < 1_00_000:
            st.caption("🟢 Small Loan — below ₹1,00,000 (easy to approve)")
        elif loan_amount < 10_00_000:
            st.caption("🟡 Medium Loan — ₹1,00,000 – ₹10,00,000")
        elif loan_amount < 50_00_000:
            st.caption("🟠 Large Loan — ₹10,00,000 – ₹50,00,000")
        else:
            st.caption("🔴 Very Large Loan — above ₹50,00,000 (harder to approve)")

        term_opts = [12, 24, 36, 48, 60, 84, 120, 180, 240, 360]
        loan_term = st.selectbox("Loan Term (months)", term_opts, index=term_opts.index(ex.get("loan_term", 60)))
        if loan_term <= 24:
            st.caption("🟢 Short Term — up to 2 years")
        elif loan_term <= 60:
            st.caption("🟡 Medium Term — 3 to 5 years")
        elif loan_term <= 120:
            st.caption("🟠 Long Term — 6 to 10 years")
        else:
            st.caption("🔵 Very Long Term — 10 to 30 years (home loans)")

        pur_opts     = ["Home", "Car", "Education", "Business", "Personal"]
        loan_purpose = st.selectbox("Loan Purpose", pur_opts, index=pur_opts.index(ex.get("loan_purpose", "Home")))
        purpose_info = {
            "Home":      "🏠 Home Loan — high value, long term, asset backed",
            "Car":       "🚗 Car Loan — medium value, asset backed",
            "Education": "🎓 Education Loan — investment in future earnings",
            "Business":  "💼 Business Loan — higher risk, variable income",
            "Personal":  "👤 Personal Loan — unsecured, higher risk",
        }
        st.caption(purpose_info[loan_purpose])

    with c2:
        area_opts     = ["Urban", "Semiurban", "Rural"]
        property_area = st.selectbox("Property Area", area_opts, index=area_opts.index(ex.get("property_area", "Urban")))
        area_info = {
            "Urban":     "🏙️ Urban — higher property value & income",
            "Semiurban": "🏘️ Semiurban — moderate value & income",
            "Rural":     "🌾 Rural — lower property value",
        }
        st.caption(area_info[property_area])

        emp_cat_opts = ["Private", "Government", "MNC", "Business", "Unemployed"]
        employer_cat = st.selectbox("Employer Category", emp_cat_opts, index=emp_cat_opts.index(ex.get("employer_cat", "Private")))
        emp_cat_info = {
            "Government": "🏛️ Government — most stable, top preference",
            "MNC":        "🌍 MNC — very stable, high salary",
            "Private":    "🏢 Private — stable but variable",
            "Business":   "💼 Business — variable income",
            "Unemployed": "🔴 Unemployed — very high risk",
        }
        st.caption(emp_cat_info[employer_cat])

        existing_loans = st.number_input("Existing Loans", 0, 20, ex.get("existing_loans", 0))
        if existing_loans == 0:
            st.caption("🟢 None — no existing loan burden")
        elif existing_loans <= 2:
            st.caption("🟡 Low — 1–2 existing loans")
        elif existing_loans <= 4:
            st.caption("🟠 Moderate — 3–4 loans, higher risk")
        else:
            st.caption("🔴 High — too many loans, very risky")
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
