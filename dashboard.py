import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(page_title="NHS Recovery Dashboard", layout="wide")

# ----------------------------
# SAFE FILE LOADING (Render Safe)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "rhs_ml_dataset.csv")

df = pd.read_csv(data_path)

# ----------------------------
# DATE PROCESSING
# ----------------------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("Filters")

if "Year" in df.columns:
    selected_year = st.sidebar.multiselect(
        "Select Year",
        options=df["Year"].dropna().unique(),
        default=df["Year"].dropna().unique()
    )
    df = df[df["Year"].isin(selected_year)]

if "Month" in df.columns:
    selected_month = st.sidebar.multiselect(
        "Select Month",
        options=df["Month"].dropna().unique(),
        default=df["Month"].dropna().unique()
    )
    df = df[df["Month"].isin(selected_month)]

if "Gender" in df.columns:
    selected_gender = st.sidebar.multiselect(
        "Select Gender",
        options=df["Gender"].dropna().unique(),
        default=df["Gender"].dropna().unique()
    )
    df = df[df["Gender"].isin(selected_gender)]

# ----------------------------
# KPI METRICS
# ----------------------------
st.title("🏥 NHS Recovery Performance Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", len(df))

if "Readmitted" in df.columns:
    readmitted_count = df["Readmitted"].sum()
    col2.metric("Total Readmissions", int(readmitted_count))

if "Age" in df.columns:
    col3.metric("Average Age", round(df["Age"].mean(), 1))

if "Gender" in df.columns:
    col4.metric("Unique Genders", df["Gender"].nunique())

st.divider()

# ----------------------------
# BAR CHART
# ----------------------------
if "Gender" in df.columns:
    gender_counts = df["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]

    fig_bar = px.bar(
        gender_counts,
        x="Gender",
        y="Count",
        title="Patients by Gender"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------
# DONUT CHART
# ----------------------------
if "Readmitted" in df.columns:
    readmit_counts = df["Readmitted"].value_counts().reset_index()
    readmit_counts.columns = ["Readmitted", "Count"]

    fig_donut = px.pie(
        readmit_counts,
        names="Readmitted",
        values="Count",
        hole=0.5,
        title="Readmission Distribution"
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ----------------------------
# FUNNEL CHART
# ----------------------------
if "Readmitted" in df.columns:
    total = len(df)
    readmitted = df["Readmitted"].sum()
    not_readmitted = total - readmitted

    fig_funnel = go.Figure(go.Funnel(
        y=["Total Patients", "Readmitted", "Not Readmitted"],
        x=[total, readmitted, not_readmitted]
    ))

    fig_funnel.update_layout(title="Patient Funnel")
    st.plotly_chart(fig_funnel, use_container_width=True)

st.divider()

# ----------------------------
# AI PREDICTION PANEL
# ----------------------------
st.header("🤖 AI Readmission Prediction")

model_path = os.path.join(BASE_DIR, "readmission_model.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder.pkl")

if os.path.exists(model_path) and os.path.exists(encoder_path):

    model = pickle.load(open(model_path, "rb"))
    encoder = pickle.load(open(encoder_path, "rb"))

    age = st.number_input("Age", min_value=0, max_value=120, value=50)

    gender = st.selectbox("Gender", df["Gender"].unique())

    if st.button("Predict Readmission Risk"):

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
    st.warning("Model files not found. Prediction panel disabled.")
