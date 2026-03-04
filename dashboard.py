import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Recovery Dashboard", layout="wide")

# -------------------------------------------------
# SECURE LOGIN
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
# LOAD DATA (CACHED)
# -------------------------------------------------
DATA_FILE = "rhs_ml_dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

if not os.path.exists(DATA_FILE):
    st.error("Dataset file not found.")
    st.stop()

df = load_data()

if df.empty:
    st.warning("Dataset is empty.")
    st.stop()

# -------------------------------------------------
# AUTO COLUMN DETECTION
# -------------------------------------------------
col_map = {c.lower(): c for c in df.columns}

age_col = col_map.get("age")
gender_col = col_map.get("gender")
readmit_col = col_map.get("readmitted") or col_map.get("readmission")
date_col = col_map.get("date") or col_map.get("admission_date")

# -------------------------------------------------
# DATE PROCESSING
# -------------------------------------------------
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Month_Name"] = df[date_col].dt.strftime("%B")
    df["Date"] = df[date_col].dt.date

filtered_df = df.copy()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

# Year Filter
if "Year" in df.columns:
    years = sorted(df["Year"].unique())
    selected_year = st.sidebar.multiselect("Year", years, default=years)
    filtered_df = filtered_df[filtered_df["Year"].isin(selected_year)]

# Month Filter
if "Month_Name" in df.columns:
    months = filtered_df["Month_Name"].unique()
    selected_month = st.sidebar.multiselect("Month", months, default=months)
    filtered_df = filtered_df[filtered_df["Month_Name"].isin(selected_month)]

# Date Range Filter
if "Date" in df.columns:
    min_date = filtered_df["Date"].min()
    max_date = filtered_df["Date"].max()
    selected_date = st.sidebar.date_input("Date Range", [min_date, max_date])

    if len(selected_date) == 2:
        filtered_df = filtered_df[
            (filtered_df["Date"] >= selected_date[0]) &
            (filtered_df["Date"] <= selected_date[1])
        ]

# Gender Filter
if gender_col:
    genders = filtered_df[gender_col].unique()
    selected_gender = st.sidebar.multiselect("Gender", genders, default=genders)
    filtered_df = filtered_df[filtered_df[gender_col].isin(selected_gender)]

# Download Button
st.sidebar.download_button(
    "Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)

# -------------------------------------------------
# DASHBOARD TITLE
# -------------------------------------------------
st.title("NHS Post-Discharge Recovery Dashboard")

# -------------------------------------------------
# KPI METRICS
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_patients = len(filtered_df)
col1.metric("Total Patients", total_patients)

if readmit_col and total_patients > 0:
    read_total = int(pd.to_numeric(filtered_df[readmit_col], errors="coerce").fillna(0).sum())
    col2.metric("Total Readmissions", read_total)
    read_rate = (read_total / total_patients) * 100
    col3.metric("Readmission Rate", f"{read_rate:.2f}%")
else:
    col2.metric("Total Readmissions", "N/A")
    col3.metric("Readmission Rate", "N/A")

# Avg Patients Per Day
if "Date" in filtered_df.columns and total_patients > 0:
    daily_counts = filtered_df.groupby("Date").size()
    avg_daily = round(daily_counts.mean(), 2)
    col4.metric("Avg Patients Per Day", avg_daily)
else:
    col4.metric("Avg Patients Per Day", "N/A")

st.divider()

# -------------------------------------------------
# CHARTS
# -------------------------------------------------

# Bar Chart - Gender
if gender_col and not filtered_df.empty:
    gender_counts = filtered_df[gender_col].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    fig_bar = px.bar(gender_counts, x="Gender", y="Count",
                     title="Patients by Gender")
    st.plotly_chart(fig_bar, use_container_width=True)

# Donut Chart - Readmission
if readmit_col and not filtered_df.empty:
    read_counts = filtered_df[readmit_col].value_counts().reset_index()
    read_counts.columns = ["Status", "Count"]
    fig_donut = px.pie(read_counts, names="Status", values="Count",
                       hole=0.5,
                       title="Readmission Distribution")
    st.plotly_chart(fig_donut, use_container_width=True)

# Daily Visits Line Chart
if "Date" in filtered_df.columns and not filtered_df.empty:
    daily_trend = filtered_df.groupby("Date").size().reset_index(name="Patients")
    fig_line = px.line(daily_trend, x="Date", y="Patients",
                       title="Daily Patient Visits")
    st.plotly_chart(fig_line, use_container_width=True)

# Funnel Chart
if readmit_col and total_patients > 0:
    read_total = int(pd.to_numeric(filtered_df[readmit_col], errors="coerce").fillna(0).sum())
    fig_funnel = go.Figure(go.Funnel(
        y=["Total Patients", "Readmitted", "Not Readmitted"],
        x=[total_patients, read_total, total_patients - read_total]
    ))
    st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# -------------------------------------------------
# AI PREDICTION PANEL
# -------------------------------------------------
MODEL_FILE = "readmission_model.pkl"
ENCODER_FILE = "encoder.pkl"

if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE) and age_col and gender_col:
    st.header("AI Readmission Prediction")

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_FILE, "rb") as f:
        encoder = pickle.load(f)

    age_input = st.number_input("Age", 0, 120, 50)
    gender_input = st.selectbox("Gender", df[gender_col].unique())

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

st.success("System Running Successfully")
