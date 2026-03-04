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
st.set_page_config(page_title="Recovery Dashboard", layout="wide")

# -------------------------------------------------
# SIMPLE LOGIN
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Secure Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "NHS2026":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# -------------------------------------------------
# LOAD DATA SAFELY
# -------------------------------------------------
DATA_FILE = "rhs_ml_dataset.csv"

if not os.path.exists(DATA_FILE):
    st.error("Dataset file not found in project directory.")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except:
    st.error("Error reading dataset.")
    st.stop()

if df.empty:
    st.error("Dataset is empty.")
    st.stop()

# -------------------------------------------------
# AUTO COLUMN DETECTION
# -------------------------------------------------
columns_lower = {c.lower(): c for c in df.columns}

age_col = columns_lower.get("age")
gender_col = columns_lower.get("gender")
readmit_col = (
    columns_lower.get("readmitted")
    or columns_lower.get("readmission")
    or columns_lower.get("readmission_flag")
)

date_col = (
    columns_lower.get("date")
    or columns_lower.get("admission_date")
    or columns_lower.get("admissiondate")
)

# -------------------------------------------------
# DATE PROCESSING
# -------------------------------------------------
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["Year"] = df[date_col].dt.year

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("NHS Recovery Performance Dashboard")

filtered_df = df.copy()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

if "Year" in df.columns:
    years = sorted(df["Year"].dropna().unique())
    selected_years = st.sidebar.multiselect("Year", years, default=years)
    filtered_df = filtered_df[filtered_df["Year"].isin(selected_years)]

if gender_col:
    genders = sorted(filtered_df[gender_col].dropna().unique())
    selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)
    filtered_df = filtered_df[filtered_df[gender_col].isin(selected_genders)]

# -------------------------------------------------
# KPIs
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_patients = len(filtered_df)
col1.metric("Total Patients", total_patients)

if readmit_col:
    read_numeric = pd.to_numeric(filtered_df[readmit_col], errors="coerce").fillna(0)
    total_readmit = int(read_numeric.sum())
    col2.metric("Total Readmissions", total_readmit)

    rate = (total_readmit / total_patients * 100) if total_patients > 0 else 0
    col3.metric("Readmission Rate", f"{rate:.2f}%")
else:
    col2.metric("Total Readmissions", "N/A")
    col3.metric("Readmission Rate", "N/A")

if age_col:
    avg_age = pd.to_numeric(filtered_df[age_col], errors="coerce").mean()
    avg_age_display = round(avg_age, 1) if not np.isnan(avg_age) else "N/A"
    col4.metric("Average Age", avg_age_display)
else:
    col4.metric("Average Age", "N/A")

st.divider()

# -------------------------------------------------
# CHARTS
# -------------------------------------------------
if gender_col:
    gender_counts = filtered_df[gender_col].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    fig1 = px.bar(gender_counts, x="Gender", y="Count", title="Patients by Gender")
    st.plotly_chart(fig1, use_container_width=True)

if readmit_col:
    read_counts = filtered_df[readmit_col].value_counts().reset_index()
    read_counts.columns = ["Status", "Count"]
    fig2 = px.pie(read_counts, names="Status", values="Count", hole=0.5,
                  title="Readmission Distribution")
    st.plotly_chart(fig2, use_container_width=True)

if date_col and date_col in filtered_df.columns:
    trend = filtered_df.groupby("Year").size().reset_index(name="Patients")
    fig3 = px.line(trend, x="Year", y="Patients", title="Yearly Patient Trend")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# -------------------------------------------------
# OPTIONAL AI PREDICTION
# -------------------------------------------------
MODEL_FILE = "readmission_model.pkl"
ENCODER_FILE = "encoder.pkl"

if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE) and age_col and gender_col:
    st.header("AI Readmission Prediction")

    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        with open(ENCODER_FILE, "rb") as f:
            encoder = pickle.load(f)

        age_input = st.number_input("Age", 0, 120, 50)
        gender_input = st.selectbox("Gender", df[gender_col].dropna().unique())

        if st.button("Predict"):
            input_df = pd.DataFrame({
                age_col: [age_input],
                gender_col: [gender_input]
            })

            encoded = encoder.transform(input_df)
            prediction = model.predict(encoded)

            if int(prediction[0]) == 1:
                st.error("High Risk of Readmission")
            else:
                st.success("Low Risk of Readmission")

    except:
        st.warning("Prediction model error.")

st.success("System Running Successfully")
