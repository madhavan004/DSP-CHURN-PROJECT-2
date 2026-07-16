import streamlit as st

from api_client import check_health

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")

st.title("📉 Telco Customer Churn Predictor")

st.markdown(
    """
Welcome! This app lets you:

- **🔮 Prediction** — make on-demand churn predictions for a single
  customer (form) or a batch of customers (CSV upload).
- **📊 Past Predictions** — browse predictions made previously, either
  from this webapp or from the scheduled Airflow prediction job.

Use the sidebar to navigate between pages.
"""
)

st.divider()

if check_health():
    st.success("✅ Model API is reachable and healthy.")
else:
    st.error(
        "❌ Model API is unreachable. Predictions and past-predictions "
        "pages will not work until the API service is back up."
    )
