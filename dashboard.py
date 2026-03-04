import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Post-Discharge Social Support and Recovery Tracker",
    layout="wide"
)

# -------------------------------------------------
# AUTHENTICATION
# -------------------------------------------------
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔐 Secure Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "NHS2026":
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "rhs_ml_dataset.csv")

if not os.path.exists(DATA_FILE):
    st.error("Dataset file not found.")
    st.stop()

df = pd.read_csv(DATA_FILE)

# -------------------------------------------------
# AUTO COLUMN DETECTION (CASE INSENSITIVE)
# -------------------------------------------------
col_map = {c.lower(): c for c in df.columns}

age_col = col_map.get("age")
gender_col = col_map.get("gender")
readmit_col = (
    col_map.get("readmitted") or
    col_map.get("readmission") or
    col_map.get("readmission_flag")
)
date_col = (
    col_map.get("date") or
    col_map.get("admission_date") or
    col_map.get("admissiondate")
)

# -------------------------------------------------
# SAFE DATE HANDLING
# -------------------------------------------------
if date_col:
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df[(df[date_col] > "2000-01-01") & (df[date_col] < "2035-01-01")]
        df["Year"] = df[date_col].dt.year
        df["Month"] = df[date_col].dt.month_name()
    except Exception:
        pass

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#002B5B;'>NHS Recovery Performance Dashboard</h1>",
    unsafe_allow_html=True
)

filtered_df = df.copy()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

if "Year" in df.columns:
    years = st.sidebar.multiselect(
        "Year",
        sorted(df["Year"].dropna().unique()),
        default=sorted(df["Year"].dropna().unique())
    )
    filtered_df = filtered_df[filtered_df["Year"].isin(years)]

if gender_col:
    genders = st.sidebar.multiselect(
        "Gender",
        sorted(filtered_df[gender_col].dropna().unique()),
        default=sorted(filtered_df[gender_col].dropna().unique())
    )
    filtered_df = filtered_df[filtered_df[gender_col].isin(genders)]

st.sidebar.download_button(
    "⬇ Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)

# -------------------------------------------------
# KPIs
# -------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Patients", len(filtered_df))

if readmit_col:
    read_numeric = pd.to_numeric(filtered_df[readmit_col], errors="coerce").fillna(0)
    read_total = int(read_numeric.sum())
    k2.metric("Total Readmissions", read_total)
    rate = (read_total / len(filtered_df) * 100) if len(filtered_df) else 0
    k3.metric("Readmission Rate", f"{rate:.2f}%")
else:
    k2.metric("Total Readmissions", "N/A")
    k3.metric("Readmission Rate", "N/A")

if age_col:
    avg_age = pd.to_numeric(filtered_df[age_col], errors="coerce").mean()
    k4.metric("Average Age", round(avg_age, 1) if not np.isnan(avg_age) else "N/A")
else:
    k4.metric("Average Age", "N/A")

st.divider()

# -------------------------------------------------
# CHARTS
# -------------------------------------------------
c1, c2 = st.columns(2)

if gender_col:
    gender_counts = filtered_df[gender_col].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    fig_bar = px.bar(
        gender_counts,
        x="Gender",
        y="Count",
        title="Patients by Gender"
    )
    c1.plotly_chart(fig_bar, use_container_width=True)

if readmit_col:
    read_counts = filtered_df[readmit_col].value_counts().reset_index()
    read_counts.columns = ["Status", "Count"]
    fig_donut = px.pie(
        read_counts,
        names="Status",
        values="Count",
        hole=0.5,
        title="Readmission Distribution"
    )
    c2.plotly_chart(fig_donut, use_container_width=True)

if date_col and date_col in filtered_df.columns:
    trend = (
        filtered_df
        .groupby(pd.Grouper(key=date_col, freq="M"))
        .size()
        .reset_index(name="Patients")
    )
    fig_trend = px.line(
        trend,
        x=date_col,
        y="Patients",
        title="Patient Trend Over Time"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

if readmit_col:
    total = len(filtered_df)
    read_total = int(pd.to_numeric(filtered_df[readmit_col], errors="coerce").fillna(0).sum())
    fig_funnel = go.Figure(go.Funnel(
        y=["Total Patients", "Readmitted", "Not Readmitted"],
        x=[total, read_total, total - read_total]
    ))
    st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# -------------------------------------------------
# AI PREDICTION PANEL
# -------------------------------------------------
st.header("🤖 AI Readmission Prediction")

MODEL = os.path.join(BASE_DIR, "readmission_model.pkl")
ENCODER = os.path.join(BASE_DIR, "encoder.pkl")

if os.path.exists(MODEL) and os.path.exists(ENCODER) and age_col and gender_col:
    try:
        with open(MODEL, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER, "rb") as f:
            encoder = pickle.load(f)

        age_input = st.number_input("Age", min_value=0, max_value=120, value=50)
        gender_input = st.selectbox(
            "Gender",
            df[gender_col].dropna().unique()
        )

        if st.button("Predict Risk"):
            input_df = pd.DataFrame({
                age_col: [age_input],
                gender_col: [gender_input]
            })
            encoded = encoder.transform(input_df)
            pred = model.predict(encoded)

            if int(pred[0]) == 1:
                st.error("High Risk of Readmission")
            else:
                st.success("Low Risk of Readmission")

    except Exception:
        st.warning("Prediction temporarily unavailable.")
else:
    st.info("Prediction model not connected.")

st.success("System Running ✔")
