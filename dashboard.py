import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="NHS Executive Recovery Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------
# AUTO REFRESH (60 seconds)
# ---------------------------------------------------
st_autorefresh(interval=60000, key="auto_refresh")

# ---------------------------------------------------
# SIMPLE AUTHENTICATION
# ---------------------------------------------------
def login():
    st.title("🔐 Executive Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "NHS2026":
            st.session_state.authenticated = True
        else:
            st.error("Invalid credentials")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login()
    st.stop()

# ---------------------------------------------------
# SAFE DATA LOADING (Render Safe)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "rhs_ml_dataset.csv")

df = pd.read_csv(data_path)

# ---------------------------------------------------
# ULTRA SAFE DATE HANDLING (Fixes OutOfBounds Error)
# ---------------------------------------------------
if "Date" in df.columns:

    try:
        # If numeric timestamps
        if np.issubdtype(df["Date"].dtype, np.number):
            df["Date"] = pd.to_datetime(df["Date"], unit="ns", errors="coerce")
        else:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Remove invalid / extreme dates
        df = df.dropna(subset=["Date"])
        df = df[
            (df["Date"] > "2000-01-01") &
            (df["Date"] < "2035-01-01")
        ]

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month_name()

    except Exception:
        st.warning("Date column contains invalid values. Trend chart disabled.")

# ---------------------------------------------------
# EXECUTIVE HEADER
# ---------------------------------------------------
st.markdown("""
<h1 style='text-align:center;color:#002B5B;'>
NHS Recovery Intelligence Dashboard
</h1>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("📊 Filters")

filtered_df = df.copy()

if "Year" in df.columns:
    years = st.sidebar.multiselect(
        "Year",
        sorted(df["Year"].unique()),
        default=sorted(df["Year"].unique())
    )
    filtered_df = filtered_df[filtered_df["Year"].isin(years)]

if "Month" in df.columns:
    months = st.sidebar.multiselect(
        "Month",
        filtered_df["Month"].unique(),
        default=filtered_df["Month"].unique()
    )
    filtered_df = filtered_df[filtered_df["Month"].isin(months)]

if "Gender" in df.columns:
    genders = st.sidebar.multiselect(
        "Gender",
        filtered_df["Gender"].unique(),
        default=filtered_df["Gender"].unique()
    )
    filtered_df = filtered_df[filtered_df["Gender"].isin(genders)]

# ---------------------------------------------------
# DOWNLOAD BUTTON
# ---------------------------------------------------
st.sidebar.download_button(
    "⬇ Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", len(filtered_df))

if "Readmitted" in filtered_df.columns:
    readmitted = filtered_df["Readmitted"].sum()
    col2.metric("Total Readmissions", int(readmitted))

if "Age" in filtered_df.columns:
    col3.metric("Average Age", round(filtered_df["Age"].mean(), 1))

if "Readmitted" in filtered_df.columns:
    rate = (readmitted / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    col4.metric("Readmission Rate", f"{round(rate,2)}%")

st.divider()

# ---------------------------------------------------
# ADVANCED ANALYTICS
# ---------------------------------------------------
colA, colB = st.columns(2)

# Gender Bar Chart
if "Gender" in filtered_df.columns:
    gender_counts = filtered_df["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]

    fig_bar = px.bar(gender_counts, x="Gender", y="Count",
                     title="Patient Distribution by Gender")
    colA.plotly_chart(fig_bar, use_container_width=True)

# Readmission Donut
if "Readmitted" in filtered_df.columns:
    readmit_counts = filtered_df["Readmitted"].value_counts().reset_index()
    readmit_counts.columns = ["Readmitted", "Count"]

    fig_donut = px.pie(readmit_counts,
                       names="Readmitted",
                       values="Count",
                       hole=0.5,
                       title="Readmission Distribution")
    colB.plotly_chart(fig_donut, use_container_width=True)

# Trend Chart
if "Date" in filtered_df.columns:
    trend = filtered_df.groupby("Date").size().reset_index(name="Patients")
    fig_trend = px.line(trend, x="Date", y="Patients",
                        title="Patient Trend Over Time")
    st.plotly_chart(fig_trend, use_container_width=True)

# Funnel
if "Readmitted" in filtered_df.columns:
    total = len(filtered_df)
    readmitted = filtered_df["Readmitted"].sum()
    not_readmitted = total - readmitted

    fig_funnel = go.Figure(go.Funnel(
        y=["Total Patients", "Readmitted", "Not Readmitted"],
        x=[total, readmitted, not_readmitted]
    ))
    fig_funnel.update_layout(title="Patient Funnel Analysis")
    st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# ---------------------------------------------------
# AI PREDICTION PANEL
# ---------------------------------------------------
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
    st.warning("Model files not found. Prediction disabled.")

st.success("Dashboard Running Successfully ✔")
