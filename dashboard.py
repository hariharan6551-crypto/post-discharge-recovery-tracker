import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")

st.title("Post Discharge Social Support & Recovery Tracker")
st.subheader("AI-Based Predictive & Recovery Monitoring Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("recovery_tracker_data.csv")

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df.dropna()

# Convert Date column
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()

# -----------------------------
# SIDEBAR FILTERS
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

# DATE FILTER
if "Date" in df.columns:
    start_date, end_date = st.sidebar.date_input(
        "Date Range",
        [filtered_df["Date"].min(), filtered_df["Date"].max()]
    )
    filtered_df = filtered_df[
        (filtered_df["Date"] >= pd.to_datetime(start_date)) &
        (filtered_df["Date"] <= pd.to_datetime(end_date))
    ]

# GENDER FILTER
if "Gender" in df.columns:
    gender = st.sidebar.multiselect(
        "Gender",
        ["Male","Female"],
        default=["Male","Female"]
    )
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

# -----------------------------
# KPI METRICS
# -----------------------------
st.header("KPI Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", len(filtered_df))

if "Recovery_Score" in filtered_df.columns:
    col2.metric("Avg Recovery Score",
                round(filtered_df["Recovery_Score"].mean(),2))

if "Persons_Visiting" in filtered_df.columns:
    col3.metric("Avg Persons Visiting",
                round(filtered_df["Persons_Visiting"].mean(),2))

if "Readmission_Risk" in filtered_df.columns:
    col4.metric("High Risk Patients",
                (filtered_df["Readmission_Risk"]=="High").sum())

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
st.header("Recovery Score by Support Level")

if "Support_Level" in filtered_df.columns and "Recovery_Score" in filtered_df.columns:

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
st.header("Support Level Distribution")

if "Support_Level" in filtered_df.columns:

    fig_donut = px.pie(
        filtered_df,
        names="Support_Level",
        hole=0.5
    )

    st.plotly_chart(fig_donut, use_container_width=True)

# -----------------------------
# FUNNEL CHART
# -----------------------------
st.header("Medication Adherence Funnel")

if "Medication_Adherence" in filtered_df.columns:

    funnel_data = filtered_df["Medication_Adherence"].value_counts().reset_index()
    funnel_data.columns = ["Stage","Count"]

    fig_funnel = px.funnel(
        funnel_data,
        x="Count",
        y="Stage"
    )

    st.plotly_chart(fig_funnel, use_container_width=True)

# -----------------------------
# YEARLY TREND
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
st.header("AI Readmission Risk Prediction")

if "Readmission_Risk" in filtered_df.columns:

    df_ml = filtered_df.copy()

    label = LabelEncoder()

    for col in df_ml.select_dtypes(include="object").columns:
        df_ml[col] = label.fit_transform(df_ml[col])

    X = df_ml.drop(columns=["Readmission_Risk"])
    y = df_ml["Readmission_Risk"]

    model = DecisionTreeClassifier()
    model.fit(X,y)

    # -----------------------------
    # REAL TIME PREDICTION PANEL
    # -----------------------------
    st.subheader("Real-Time Patient Risk Prediction")

    input_data = {}

    for col in X.columns:
        val = st.number_input(f"{col}", value=1)
        input_data[col] = val

    if st.button("Predict Risk"):

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)

        st.success(f"Predicted Risk Level: {prediction[0]}")
