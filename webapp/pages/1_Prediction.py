import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

from api_client import APIError, predict
from schema_constants import CATEGORICAL_OPTIONS, REQUIRED_COLUMNS

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Make a Prediction")

# Results live in session_state so they survive Streamlit reruns.
if "single_prediction_result" not in st.session_state:
    st.session_state.single_prediction_result = None
if "batch_prediction_result" not in st.session_state:
    st.session_state.batch_prediction_result = None


def _render_single_result(pred: dict):
    """Render a prominent result card for a single prediction."""
    churn = pred["prediction"]
    prob = pred["churn_probability"]
    cust_id = pred.get("customer_id") or "N/A"

    if churn == "Yes":
        color, bg, border, icon, label = "#c0392b", "#fff0f0", "#e74c3c", "🔴", "WILL CHURN"
    else:
        color, bg, border, icon, label = "#27ae60", "#f0fff4", "#2ecc71", "🟢", "WILL NOT CHURN"

    bar_color = "#e74c3c" if prob >= 0.5 else "#2ecc71"
    bar_pct = int(prob * 100)

    st.markdown(
        f"""
        <div style="background:{bg};border:2px solid {border};border-radius:12px;
                    padding:24px 32px;margin:16px 0;">
            <div style="font-size:2.2rem;font-weight:800;color:{color};">{icon} {label}</div>
            <div style="font-size:1.1rem;color:#555;margin-top:6px;">
                Customer: <strong>{cust_id}</strong> &nbsp;|&nbsp;
                Model: <strong>{pred.get("model_version", "—")}</strong>
            </div>
            <div style="margin-top:16px;">
                <div style="font-size:0.9rem;color:#888;margin-bottom:4px;">
                    Churn Probability: <strong style="color:{color}">{prob:.1%}</strong>
                </div>
                <div style="background:#e0e0e0;border-radius:8px;height:18px;width:100%;">
                    <div style="width:{bar_pct}%;background:{bar_color};height:18px;
                                border-radius:8px;"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _color_pred(val):
    if val == "Yes":
        return "background-color:#ffe0e0;color:#c0392b;font-weight:bold"
    return "background-color:#e0ffe4;color:#27ae60;font-weight:bold"


# ---------------------------------------------------------------------------
# Tabs — ONLY contain inputs and submit buttons, NOT results
# ---------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["Single prediction (form)", "Batch prediction (CSV upload)"])

with tab_single:
    st.subheader("Enter customer details")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            customer_id = st.text_input("Customer ID (optional)", value="")
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Has Partner", CATEGORICAL_OPTIONS["Partner"])
            dependents = st.selectbox("Has Dependents", CATEGORICAL_OPTIONS["Dependents"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

        with col2:
            phone_service = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["PhoneService"])
            multiple_lines = st.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["MultipleLines"])
            internet_service = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["InternetService"])
            online_security = st.selectbox("Online Security", CATEGORICAL_OPTIONS["OnlineSecurity"])
            online_backup = st.selectbox("Online Backup", CATEGORICAL_OPTIONS["OnlineBackup"])
            device_protection = st.selectbox("Device Protection", CATEGORICAL_OPTIONS["DeviceProtection"])

        with col3:
            tech_support = st.selectbox("Tech Support", CATEGORICAL_OPTIONS["TechSupport"])
            streaming_tv = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])
            streaming_movies = st.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"])
            contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
            paperless_billing = st.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["PaperlessBilling"])
            payment_method = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["PaymentMethod"])

        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=50.0
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=100000.0, value=600.0
        )

        submitted = st.form_submit_button("🔮 Predict", type="primary")

    # Call API immediately when form is submitted (still inside tab_single
    # so we can access the form values, but result is rendered BELOW the tabs)
    if submitted:
        st.session_state.active_result_tab = "single"
        features = {
            "customerID": customer_id or None,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        try:
            with st.spinner("Calling model API…"):
                result = predict([features], source="webapp")
        except APIError as e:
            st.session_state.single_prediction_result = ("error", str(e))
        else:
            preds = result.get("predictions", [])
            if not preds:
                st.session_state.single_prediction_result = ("empty", None)
            else:
                st.session_state.single_prediction_result = ("ok", preds[0])

with tab_batch:
    st.subheader("Upload a CSV file for batch predictions")
    st.caption(
        "The file must contain the feature columns below, "
        "**without** the Churn/label column: "
        + ", ".join(REQUIRED_COLUMNS)
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            df_upload = None

        if df_upload is not None:
            missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_upload.columns]
            if missing_cols:
                st.error("Missing required columns: " + ", ".join(missing_cols))
            elif df_upload.empty:
                st.info("The uploaded file has no rows.")
            else:
                st.write(f"Loaded **{len(df_upload)}** rows. Preview:")
                st.dataframe(df_upload.head(10), use_container_width=True)

                if st.button("Run batch prediction", type="primary"):
                    st.session_state.active_result_tab = "batch"
                    records = df_upload.to_dict(orient="records")
                    try:
                        with st.spinner(f"Predicting {len(records)} rows…"):
                            result = predict(records, source="webapp")
                    except APIError as e:
                        st.session_state.batch_prediction_result = ("error", str(e))
                    else:
                        preds = result.get("predictions", [])
                        if not preds:
                            st.session_state.batch_prediction_result = ("empty", None)
                        else:
                            rows = []
                            for p in preds:
                                row = {"customer_id": p.get("customer_id") or "—",
                                       "prediction": p["prediction"],
                                       "churn_probability": p["churn_probability"]}
                                rows.append(row)
                            st.session_state.batch_prediction_result = ("ok", pd.DataFrame(rows))

# ---------------------------------------------------------------------------
# Result display — OUTSIDE all tabs so it always renders after any submission
# ---------------------------------------------------------------------------
st.divider()

active = st.session_state.get("active_result_tab")

if active == "single" and st.session_state.single_prediction_result is not None:
    kind, payload = st.session_state.single_prediction_result
    if kind == "error":
        st.error(f"❌ Prediction failed: {payload}")
    elif kind == "empty":
        st.info("The API returned no predictions.")
    else:
        st.markdown("### 🎯 Prediction Result")
        _render_single_result(payload)

elif active == "batch" and st.session_state.batch_prediction_result is not None:
    kind, payload = st.session_state.batch_prediction_result
    if kind == "error":
        st.error(f"❌ Batch prediction failed: {payload}")
    elif kind == "empty":
        st.info("The API returned no predictions.")
    else:
        total = len(payload)
        churn_yes = int((payload["prediction"] == "Yes").sum())
        churn_no = int((payload["prediction"] == "No").sum())
        total = len(payload)

        st.markdown("### 🎯 Batch Prediction Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Rows", total)
        m2.metric("🔴 Churn = Yes", churn_yes)
        m3.metric("🟢 Churn = No", churn_no)

        styled = payload.style.map(_color_pred, subset=["prediction"])
        st.dataframe(styled, use_container_width=True, height=400)
