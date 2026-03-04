import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")

st.title("Post Discharge Social Support & Recovery Tracker")
st.subheader("AI-Based Predictive & Recovery Monitoring Dashboard")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("recovery_tracker_data.csv")

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.dropna()

# Convert date column if exists
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

# Year filter
if "Year" in df.columns:
    year_filter = st.sidebar.multiselect(
        "Select Year",
        options=df["Year"].unique(),
        default=df["Year"].unique()
    )
    df = df[df["Year"].isin(year_filter)]

# Gender filter
if "Gender" in df.columns:
    gender_filter = st.sidebar.multiselect(
        "Select Gender",
        options=df["Gender"].unique(),
        default=df["Gender"].unique()
    )
    df = df[df["Gender"].isin(gender_filter)]

# Month filter
if "Month" in df.columns:
    month_filter = st.sidebar.multiselect(
        "Select Month",
        options=df["Month"].unique(),
        default=df["Month"].unique()
    )
    df = df[df["Month"].isin(month_filter)]

# Date filter
if "Date" in df.columns:
    date_filter = st.sidebar.date_input(
        "Select Date Range",
        [df["Date"].min(), df["Date"].max()]
    )
    df = df[(df["Date"] >= pd.to_datetime(date_filter[0])) &
            (df["Date"] <= pd.to_datetime(date_filter[1]))]

# -----------------------------
# KPI Metrics
# -----------------------------
st.header("KPI Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Patients", len(df))

if "Recovery_Score" in df.columns:
    col2.metric("Avg Recovery Score", round(df["Recovery_Score"].mean(),2))

if "Readmission_Risk" in df.columns:
    col3.metric("High Risk Patients",
                (df["Readmission_Risk"]=="High").sum())

# -----------------------------
# Aggregated Dataset
# -----------------------------
st.header("Aggregated Dataset")

if "Support_Level" in df.columns:
    agg = df.groupby("Support_Level").mean(numeric_only=True)
    st.dataframe(agg)

# -----------------------------
# Bar Chart
# -----------------------------
st.header("Recovery Score by Support Level")

if "Support_Level" in df.columns and "Recovery_Score" in df.columns:

    fig_bar = px.bar(
        df,
        x="Support_Level",
        y="Recovery_Score",
        color="Support_Level"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Donut Chart
# -----------------------------
st.header("Support Level Distribution")

if "Support_Level" in df.columns:

    fig_donut = px.pie(
        df,
        names="Support_Level",
        hole=0.5
    )

    st.plotly_chart(fig_donut, use_container_width=True)

# -----------------------------
# Funnel Chart
# -----------------------------
st.header("Medication Adherence Funnel")

if "Medication_Adherence" in df.columns:

    funnel_data = df["Medication_Adherence"].value_counts().reset_index()
    funnel_data.columns = ["Stage","Count"]

    fig_funnel = px.funnel(
        funnel_data,
        x="Count",
        y="Stage"
    )

    st.plotly_chart(fig_funnel, use_container_width=True)

# -----------------------------
# Yearly Trend Chart
# -----------------------------
if "Year" in df.columns and "Recovery_Score" in df.columns:

    st.header("Yearly Recovery Trend")

    fig_year = px.bar(
        df,
        x="Year",
        y="Recovery_Score",
        color="Support_Level"
    )

    st.plotly_chart(fig_year, use_container_width=True)

# -----------------------------
# MACHINE LEARNING MODEL
# -----------------------------
st.header("AI Readmission Risk Prediction")

if "Readmission_Risk" in df.columns:

    df_ml = df.copy()

    label = LabelEncoder()

    for col in df_ml.select_dtypes(include="object").columns:
        df_ml[col] = label.fit_transform(df_ml[col])

    X = df_ml.drop(columns=["Readmission_Risk"])
    y = df_ml["Readmission_Risk"]

    model = DecisionTreeClassifier()
    model.fit(X,y)

    # -----------------------------
    # Prediction Panel
    # -----------------------------
    st.subheader("Real-Time Patient Risk Prediction")

    input_data = {}

    for col in X.columns:
        val = st.number_input(f"Enter {col}", value=1)
        input_data[col] = val

    if st.button("Predict Risk"):

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)

        st.success(f"Predicted Risk Level: {prediction[0]}")
