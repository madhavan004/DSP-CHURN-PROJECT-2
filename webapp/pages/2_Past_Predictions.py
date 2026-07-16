import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

from api_client import APIError, get_past_predictions

st.set_page_config(page_title="Past Predictions", page_icon="📊", layout="wide")
st.title("📊 Past Predictions")

if "past_predictions_result" not in st.session_state:
    st.session_state.past_predictions_result = None

# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
st.markdown("### Filters")
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    default_start = datetime.now(timezone.utc) - timedelta(days=7)
    date_range = st.date_input(
        "Date range",
        value=(default_start.date(), datetime.now(timezone.utc).date()),
    )

with col2:
    source = st.selectbox("Source", ["all", "webapp", "scheduled"])

with col3:
    limit = st.number_input("Max rows", min_value=10, max_value=5000, value=500, step=10)

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    load_btn = st.button("🔍 Load / Refresh", type="primary", use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Fetch only when button is clicked
# --------------------------------------------------------------------------
if load_btn:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
    else:
        start_date, end_date = None, None

    try:
        with st.spinner("Fetching predictions…"):
            result = get_past_predictions(
                start_date=start_date, end_date=end_date,
                source=source, limit=int(limit),
            )
        predictions = result.get("predictions", [])
        total_count = result.get("count", 0)

        if not predictions:
            st.session_state.past_predictions_result = ("empty", None)
        else:
            rows = []
            for p in predictions:
                rows.append({
                    "ID": p["id"],
                    "Customer ID": p["customer_id"] or "—",
                    "Prediction": p["prediction"],
                    "Churn Prob.": round(p["churn_probability"], 4),
                    "Model": p["model_version"],
                    "Source": p["source"],
                    "Timestamp": p["created_at"],
                })
            st.session_state.past_predictions_result = (
                "ok", (total_count, len(predictions), pd.DataFrame(rows))
            )
    except APIError as e:
        st.session_state.past_predictions_result = ("error", str(e))

# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
result = st.session_state.past_predictions_result

if result is None:
    st.info("👆 Set your filters above and click **Load / Refresh** to view predictions.")

else:
    kind, payload = result

    if kind == "error":
        st.error(f"❌ Could not fetch past predictions: {payload}")

    elif kind == "empty":
        st.warning("No predictions found for the selected filters.")

    else:
        total_count, shown_count, df = payload

        # Summary metrics
        churn_yes = int((df["Prediction"] == "Yes").sum())
        churn_no  = int((df["Prediction"] == "No").sum())
        churn_rate = churn_yes / shown_count * 100 if shown_count else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total in DB",   total_count)
        m2.metric("Showing",       shown_count)
        m3.metric("🔴 Churn = Yes", churn_yes)
        m4.metric("🟢 Churn = No",  churn_no)
        m5.metric("Churn Rate",    f"{churn_rate:.1f}%")

        st.divider()

        # Colour-coded table
        def _color_pred(val):
            if val == "Yes":
                return "background-color:#ffe0e0;color:#c0392b;font-weight:bold"
            return "background-color:#e0ffe4;color:#27ae60;font-weight:bold"

        def _color_prob(val):
            if val >= 0.7:
                return "color:#c0392b;font-weight:bold"
            if val >= 0.4:
                return "color:#e67e22;font-weight:bold"
            return "color:#27ae60"

        styled = (
            df.style
            .map(_color_pred, subset=["Prediction"])
            .map(_color_prob, subset=["Churn Prob."])
            .format({"Churn Prob.": "{:.2%}"})
        )

        st.dataframe(styled, use_container_width=True, height=500)
