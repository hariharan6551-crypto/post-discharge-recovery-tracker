import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from streamlit_autorefresh import st_autorefresh

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="NHS Executive Recovery Dashboard",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------------
# AUTO REFRESH (Every 60 seconds)
# -----------------------------------
st_autorefresh(interval=60000, key="refresh")

# -----------------------------------
# SIMPLE AUTHENTICATION
# -----------------------------------
def login():
    st.title("🔐 Executive Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "NHS2026":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid credentials")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# -----------------------------------
# LOAD DATA SAFELY (Render safe)
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "rhs_ml_dataset.csv")
df = pd.read_csv(data_path)

# -----------------------------------
# DATE PROCESSING
# -----------------------------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()

# -----------------------------------
# EXECUTIVE HEADER
# -----------------------------------
st.markdown("""
    <h1 style='text-align:center;color:#003366;'>
    NHS Recovery Intelligence Dashboard
    </h1>
""", unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR FILTERS
# -----------------------------------
st.sidebar.header("📊 Filters")

if "Year" in df.columns:
    years = st.sidebar.multiselect(
        "Year",
        df["Year"].dropna().unique(),
        default=df["Year"].dropna().unique()
    )
    df = df[df["Year"].isin(years)]

if "Month" in df.columns:
    months = st.sidebar.multiselect(
        "Month",
        df["Month"].dropna().unique(),
        default=df["Month"].dropna().unique()
    )
    df = df[df["Month"].isin(months)]

if "Gender" in df.columns:
    genders = st.sidebar.multiselect(
        "Gender",
        df["Gender"].dropna().unique(),
        default=df["Gender"].dropna().unique()
    )
    df = df[df["Gender"].isin(genders)]

st.sidebar.divider()

# -----------------------------------
# DOWNLOAD BUTTONS
# -----------------------------------
st.sidebar.download_button(
    label="⬇ Download Filtered Data",
    data=df.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)

# -----------------------------------
# KPI SECTION
# -----------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", len(df))

if "Readmitted" in df.columns:
    readmitted = df["Readmitted"].sum()
    col2.metric("Total Readmissions", int(readmitted))

if "Age" in df.columns:
    col3.metric("Average Age", round(df["Age"].mean(), 1))

if "Readmitted" in df.columns:
    rate = (readmitted / len(df)) * 100 if len(df) > 0 else 0
    col4.metric("Readmission Rate", f"{round(rate,2)}%")

st.divider()

# -----------------------------------
# ADVANCED ANALYTICS SECTION
# -----------------------------------
colA, colB = st.columns(2)

# Bar Chart
if "Gender" in df.columns:
    gender_counts = df["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]

    fig_bar = px.bar(gender_counts, x="Gender", y="Count",
                     title="Patient Distribution by Gender")
    colA.plotly_chart(fig_bar, use_container_width=True)

# Donut Chart
if "Readmitted" in df.columns:
    readmit_counts = df["Readmitted"].value_counts().reset_index()
    readmit_counts.columns = ["Readmitted", "Count"]

    fig_donut = px.pie(readmit_counts,
                       names="Readmitted",
                       values="Count",
                       hole=0.5,
                       title="Readmission Distribution")
    colB.plotly_chart(fig_donut, use_container_width=True)

# Trend Analysis
if "Date" in df.columns:
    trend = df.groupby("Date").size().reset_index(name="Patients")
    fig_trend = px.line(trend, x="Date", y="Patients",
                        title="Patient Trend Over Time")
    st.plotly_chart(fig_trend, use_container_width=True)

# Funnel Chart
if "Readmitted" in df.columns:
    total = len(df)
    readmitted = df["Readmitted"].sum()
    not_readmitted = total - readmitted

    fig_funnel = go.Figure(go.Funnel(
        y=["Total Patients", "Readmitted", "Not Readmitted"],
        x=[total, readmitted, not_readmitted]
    ))
    fig_funnel.update_layout(title="Patient Funnel Analysis")
    st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# -----------------------------------
# AI PREDICTION PANEL
# -----------------------------------
st.header("🤖 AI Readmission Prediction")

model_path = os.path.join(BASE_DIR, "readmission_model.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder.pkl")

if os.path.exists(model_path) and os.path.exists(encoder_path):

    model = pickle.load(open(model_path, "rb"))
    encoder = pickle.load(open(encoder_path, "rb"))

    age = st.number_input("Age", 0, 120, 50)
    gender = st.selectbox("Gender", df["Gender"].unique())

    if st.button("Predict Risk"):

        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender]
        })

        input_encoded = encoder.transform(input_df)
        prediction = model.predict(input_encoded)

        if prediction[0] == 1:
            st.error("⚠ High Risk of Readmission")
        else:
            st.success("✅ Low Risk of Readmission")

else:
    st.warning("Model files not found.")

st.success("Dashboard Running Successfully ✔")
