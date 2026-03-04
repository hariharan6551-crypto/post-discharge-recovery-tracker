import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Recovery Dashboard",
    layout="wide"
)

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
# LOAD DATA
# -------------------------------------------------
DATA_FILE = "rhs_ml_dataset.csv"

if not os.path.exists(DATA_FILE):
    st.error("Dataset file not found.")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except Exception:
    st.error("Error reading dataset.")
    st.stop()

if df.empty:
    st.warning("Dataset is empty.")
    st.stop()

# -------------------------------------------------
# AUTO COLUMN DETECTION
# -------------------------------------------------
col_map = {c.lower(): c for c in df.columns}

age_col = col_map.get("age")
gender_col = col_map.get("gender")
readmit_col = col_map.get("readmitted") or col_map.get("readmission") or col_map.get("readmission_flag")
date_col = col_map.get("date") or col_map.get("admission_date") or col_map.get("admissiondate")

# -------------------------------------------------
# DATE PROCESSING
# -------------------------------------------------
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if not df.empty:
        df["Year"] = df[date_col].dt.year

filtered_df = df.copy()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

if "Year" in df.columns:
    years = sorted(df["Year"].dropna().unique())
    if years:
        selected_years = st.sidebar.multiselect("Year", years, default=years)
        filtered_df = filtered_df[filtered_df["Year"].isin(selected_years)]

if gender_col:
    genders = sorted(filtered_df[gender_col].dropna().unique())
    if genders:
        selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)
        filtered_df = filtered_df[filtered_df[gender_col].isin(selected_genders)]

st.sidebar.download_button(
    "Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("NHS Recovery Performance Dashboard")

# -------------------------------------------------
# KPIs
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_patients = len(filtered_df)
col1.metric("Total Patients", total_patients)

if readmit_col and total_patients > 0:
    read_total = int(pd.to_numeric(filtered_df[readmit_col], errors="coerce").fillna(0).sum())
    col2.metric("Total Readmissions", read_total)
    rate = (read_total / total_patients) * 100
    col3.metric("Readmission Rate", f"{rate:.2f}%")
else:
    col2.metric("Total Readmissions", "N/A")
    col3.metric("Readmission Rate", "N/A")

if age_col and total_patients > 0:
    avg_age = pd.to_numeric(filtered_df[age_col], errors="coerce").mean()
    col4.metric("Average Age", round(avg_age, 1))
else:
    col4.metric("Average Age", "N/A")

st.divider()

# -------------------------------------------------
# CHARTS
# -------------------------------------------------
if gender_col and not filtered_df.empty:
    gender_counts = filtered_df[gender_col].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    fig1 = px.bar(gender_counts, x="Gender", y="Count", title="Patients by Gender")
    st.plotly_chart(fig1, use_container_width=True)

if readmit_col and not filtered_df.empty:
    read_counts = filtered_df[readmit_col].value_counts().reset_index()
    read_counts.columns = ["Status", "Count"]
    fig2 = px.pie(read_counts, names="Status", values="Count", hole=0.5,
                  title="Readmission Distribution")
    st.plotly_chart(fig2, use_container_width=True)

if date_col and "Year" in filtered_df.columns and not filtered_df.empty:
    trend = filtered_df.groupby("Year").size().reset_index(name="Patients")
    fig3 = px.line(trend, x="Year", y="Patients", title="Yearly Patient Trend")
    st.plotly_chart(fig3, use_container_width=True)

if readmit_col and total_patients > 0:
    read_total = int(pd.to_numeric(filtered_df[readmit_col], errors="coerce").fillna(0).sum())
    fig4 = go.Figure(go.Funnel(
        y=["Total Patients", "Readmitted", "Not Readmitted"],
        x=[total_patients, read_total, total_patients - read_total]
    ))
    st.plotly_chart(fig4, use_container_width=True)

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
        gender_options = df[gender_col].dropna().unique()

        if len(gender_options) > 0:
            gender_input = st.selectbox("Gender", gender_options)

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
    except Exception:
        st.warning("Prediction unavailable.")

st.success("System Running Successfully")
