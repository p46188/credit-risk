
import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("credit_default_model.pkl")
scaler = joblib.load("credit_scaler.pkl")
features = joblib.load("credit_features.pkl")

st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Default Risk Prediction")
st.write(
    "Decision-support tool using Logistic Regression to estimate "
    "a customer's probability of credit-card default."
)

st.info(
    "This model is intended for risk screening and decision support, "
    "not automatic credit approval or rejection."
)

# -------------------------
# CUSTOMER PROFILE
# -------------------------

st.subheader("1. Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    LIMIT_BAL = st.number_input(
        "Credit Limit",
        min_value=0,
        value=200000,
        step=10000
    )

    AGE = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

with col2:
    SEX = st.selectbox(
        "Sex",
        options=[1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female"
    )

    EDUCATION = st.selectbox(
        "Education",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Graduate School",
            2: "University",
            3: "High School",
            4: "Other"
        }[x]
    )

with col3:
    MARRIAGE = st.selectbox(
        "Marital Status",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Married",
            2: "Single",
            3: "Other"
        }[x]
    )


# -------------------------
# REPAYMENT HISTORY
# -------------------------

st.subheader("2. Repayment Status")

st.caption(
    "Negative values generally indicate early/no outstanding payment; "
    "positive values represent repayment delay."
)

pay1, pay2, pay3 = st.columns(3)

with pay1:
    PAY_0 = st.number_input("Latest Month Status (PAY_0)", -2, 8, 0)
    PAY_2 = st.number_input("2 Months Ago (PAY_2)", -2, 8, 0)

with pay2:
    PAY_3 = st.number_input("3 Months Ago (PAY_3)", -2, 8, 0)
    PAY_4 = st.number_input("4 Months Ago (PAY_4)", -2, 8, 0)

with pay3:
    PAY_5 = st.number_input("5 Months Ago (PAY_5)", -2, 8, 0)
    PAY_6 = st.number_input("6 Months Ago (PAY_6)", -2, 8, 0)


# -------------------------
# BILL AMOUNTS
# -------------------------

st.subheader("3. Monthly Bill Amounts")

b1, b2, b3 = st.columns(3)

with b1:
    BILL_AMT1 = st.number_input("Bill Amount 1", value=50000)
    BILL_AMT2 = st.number_input("Bill Amount 2", value=48000)

with b2:
    BILL_AMT3 = st.number_input("Bill Amount 3", value=45000)
    BILL_AMT4 = st.number_input("Bill Amount 4", value=42000)

with b3:
    BILL_AMT5 = st.number_input("Bill Amount 5", value=40000)
    BILL_AMT6 = st.number_input("Bill Amount 6", value=38000)


# -------------------------
# PAYMENT AMOUNTS
# -------------------------

st.subheader("4. Previous Payment Amounts")

p1, p2, p3 = st.columns(3)

with p1:
    PAY_AMT1 = st.number_input("Payment Amount 1", min_value=0, value=5000)
    PAY_AMT2 = st.number_input("Payment Amount 2", min_value=0, value=5000)

with p2:
    PAY_AMT3 = st.number_input("Payment Amount 3", min_value=0, value=5000)
    PAY_AMT4 = st.number_input("Payment Amount 4", min_value=0, value=5000)

with p3:
    PAY_AMT5 = st.number_input("Payment Amount 5", min_value=0, value=5000)
    PAY_AMT6 = st.number_input("Payment Amount 6", min_value=0, value=5000)


# -------------------------
# CREATE INPUT
# -------------------------

customer_data = {
    "LIMIT_BAL": LIMIT_BAL,
    "SEX": SEX,
    "EDUCATION": EDUCATION,
    "MARRIAGE": MARRIAGE,
    "AGE": AGE,

    "PAY_0": PAY_0,
    "PAY_2": PAY_2,
    "PAY_3": PAY_3,
    "PAY_4": PAY_4,
    "PAY_5": PAY_5,
    "PAY_6": PAY_6,

    "BILL_AMT1": BILL_AMT1,
    "BILL_AMT2": BILL_AMT2,
    "BILL_AMT3": BILL_AMT3,
    "BILL_AMT4": BILL_AMT4,
    "BILL_AMT5": BILL_AMT5,
    "BILL_AMT6": BILL_AMT6,

    "PAY_AMT1": PAY_AMT1,
    "PAY_AMT2": PAY_AMT2,
    "PAY_AMT3": PAY_AMT3,
    "PAY_AMT4": PAY_AMT4,
    "PAY_AMT5": PAY_AMT5,
    "PAY_AMT6": PAY_AMT6
}

input_df = pd.DataFrame([customer_data])

# Ensure same feature order as training data
input_df = input_df[features]


# -------------------------
# PREDICTION
# -------------------------

st.divider()

if st.button("Predict Default Risk", type="primary"):

    scaled_data = scaler.transform(input_df)

    probability = model.predict_proba(scaled_data)[0][1]

    # Keep current model's standard threshold
    threshold = 0.30

    st.subheader("Risk Assessment")

    st.metric(
        "Estimated Probability of Default",
        f"{probability * 100:.2f}%"
    )

    if probability >= threshold:

        st.error("⚠️ Higher Default Risk")

        st.write(
            "The customer has been flagged for additional credit-risk review."
        )

    else:

        st.success("✅ Lower Default Risk")

        st.write(
            "The model currently classifies this customer as lower default risk."
        )

    # Risk band
    if probability < 0.30:
        risk_band = "Low"
    elif probability < 0.50:
        risk_band = "Moderate"
    elif probability < 0.70:
        risk_band = "High"
    else:
        risk_band = "Very High"

    st.write("### Risk Band:", risk_band)

    st.caption(
        "Prediction should be combined with institutional credit policy, "
        "verification and human review."
    )
