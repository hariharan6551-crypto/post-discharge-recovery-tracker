import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Recovery Tracker Dashboard", layout="wide")

st.title("Post Discharge Social Support & Recovery Tracker")
st.subheader("AI-Based Predictive & Recovery Monitoring Dashboard")

# -----------------------------
# LOAD DATASET
# -----------------------------

file_path = "rhs_ml_dataset.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    st.error("Dataset file 'rhs_ml_dataset.csv' not found in project folder.")
    st.stop()

df = df.dropna()

# -----------------------------
# DATE PROCESSING
# -----------------------------

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()

# -----------------------------
# FILTERS
# -----------------------------

st.sidebar.header("Filters")

filtered_df = df.copy()

# YEAR FILTER
if "Year" in df.columns:
    year = st.sidebar.multiselect(
        "Year",
        sorted(df["Year"].unique()),
        default=sorted(df["Year"].unique())
    )
    filtered_df = filtered_df[filtered_df["Year"].isin(year)]

# MONTH FILTER
if "Month" in df.columns:
    month = st.sidebar.multiselect(
        "Month",
        filtered_df["Month"].unique(),
        default=filtered_df["Month"].unique()
    )
    filtered_df = filtered_df[filtered_df["Month"].isin(month)]

# DATE RANGE
if "Date" in df.columns:
    start, end = st.sidebar.date_input(
        "Date Range",
        [filtered_df["Date"].min(), filtered_df["Date"].max()]
    )

    filtered_df = filtered_df[
        (filtered_df["Date"] >= pd.to_datetime(start)) &
        (filtered_df["Date"] <= pd.to_datetime(end))
    ]

# GENDER FILTER
if "Gender" in df.columns:
    gender = st.sidebar.multiselect(
        "Gender",
        filtered_df["Gender"].unique(),
        default=filtered_df["Gender"].unique()
    )
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

# -----------------------------
# KPI METRICS
# -----------------------------

st.header("KPI Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", len(filtered_df))

if "Recovery_Score" in filtered_df.columns:
    col2.metric(
        "Average Recovery Score",
        round(filtered_df["Recovery_Score"].mean(), 2)
    )

if "Persons_Visiting" in filtered_df.columns:
    col3.metric(
        "Avg Persons Visiting",
        round(filtered_df["Persons_Visiting"].mean(), 2)
    )

if "Readmission_Risk" in filtered_df.columns:
    col4.metric(
        "High Risk Patients",
        (filtered_df["Readmission_Risk"] == "High").sum()
    )

# -----------------------------
# AGGREGATED DATASET
# -----------------------------

st.header("Aggregated Dataset")

if "Support_Level" in filtered_df.columns:
    agg = filtered_df.groupby("Support_Level").mean(numeric_only=True)
    st.dataframe(agg)

# -----------------------------
# BAR CHART
# -----------------------------

if "Support_Level" in filtered_df.columns and "Recovery_Score" in filtered_df.columns:

    st.header("Recovery Score by Support Level")

    fig_bar = px.bar(
        filtered_df,
        x="Support_Level",
        y="Recovery_Score",
        color="Support_Level"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# DONUT CHART
# -----------------------------

if "Support_Level" in filtered_df.columns:

    st.header("Support Level Distribution")

    fig_donut = px.pie(
        filtered_df,
        names="Support_Level",
        hole=0.5
    )

    st.plotly_chart(fig_donut, use_container_width=True)

# -----------------------------
# FUNNEL CHART
# -----------------------------

if "Medication_Adherence" in filtered_df.columns:

    st.header("Medication Adherence Funnel")

    funnel_data = filtered_df["Medication_Adherence"].value_counts().reset_index()
    funnel_data.columns = ["Stage", "Count"]

    fig_funnel = go.Figure(go.Funnel(
        y=funnel_data["Stage"],
        x=funnel_data["Count"]
    ))

    st.plotly_chart(fig_funnel, use_container_width=True)

# -----------------------------
# YEAR TREND CHART
# -----------------------------

if "Year" in filtered_df.columns and "Recovery_Score" in filtered_df.columns:

    st.header("Yearly Recovery Trend")

    fig_year = px.bar(
        filtered_df,
        x="Year",
        y="Recovery_Score",
        color="Support_Level"
    )

    st.plotly_chart(fig_year, use_container_width=True)

# -----------------------------
# MACHINE LEARNING MODEL
# -----------------------------

if "Readmission_Risk" in filtered_df.columns:

    st.header("AI Predictive Model")

    df_ml = filtered_df.copy()

    label = LabelEncoder()

    for col in df_ml.select_dtypes(include="object").columns:
        df_ml[col] = label.fit_transform(df_ml[col])

    X = df_ml.drop(columns=["Readmission_Risk"])
    y = df_ml["Readmission_Risk"]

    model = DecisionTreeClassifier()
    model.fit(X, y)

    # -----------------------------
    # REAL TIME PREDICTION
    # -----------------------------

    st.subheader("Real-Time Risk Prediction")

    input_data = {}

    for col in X.columns:
        val = st.number_input(col, value=1)
        input_data[col] = val

    if st.button("Predict Risk"):

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)

        st.success(f"Predicted Risk Level: {prediction[0]}")
