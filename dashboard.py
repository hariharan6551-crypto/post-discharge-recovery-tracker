import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Post-Discharge Social Support and Recovery Tracker",
    layout="wide"
)

# -------------------------------------------------
# LOGIN
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
# LOAD DATA (MATCHES YOUR FILE EXACTLY)
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("rhs_ml_dataset.csv", low_memory=False)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Extract numeric year (2014 from 2014/15)
    df["Year_numeric"] = df["Year"].str[:4].astype(int)

    # Convert numeric columns safely
    df["Indicator value"] = pd.to_numeric(df["Indicator value"], errors="coerce")
    df["Numerator"] = pd.to_numeric(df["Numerator"].astype(str).str.replace(",", ""), errors="coerce")
    df["Denominator"] = pd.to_numeric(df["Denominator"].astype(str).str.replace(",", ""), errors="coerce")

    return df

df = load_data()

st.title("Post-Discharge Social Support and Recovery Tracker")

# -------------------------------------------------
# FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

year_filter = st.sidebar.multiselect(
    "Select Year",
    sorted(df["Year_numeric"].unique()),
    default=sorted(df["Year_numeric"].unique())
)

sex_filter = st.sidebar.multiselect(
    "Select Sex",
    df["Sex Breakdown"].unique(),
    default=df["Sex Breakdown"].unique()
)

filtered_df = df[
    (df["Year_numeric"].isin(year_filter)) &
    (df["Sex Breakdown"].isin(sex_filter))
]

# -------------------------------------------------
# KPI METRICS
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Average Indicator Value", round(filtered_df["Indicator value"].mean(), 2))
col2.metric("Total Numerator", int(filtered_df["Numerator"].sum()))
col3.metric("Total Denominator", int(filtered_df["Denominator"].sum()))

st.divider()

# -------------------------------------------------
# BAR CHART (Indicator by Year)
# -------------------------------------------------
bar_data = filtered_df.groupby("Year_numeric")["Indicator value"].mean().reset_index()

bar = px.bar(
    bar_data,
    x="Year_numeric",
    y="Indicator value",
    title="Average Indicator Value by Year"
)

st.plotly_chart(bar, use_container_width=True)

# -------------------------------------------------
# DONUT CHART (Sex Distribution by Numerator)
# -------------------------------------------------
donut_data = filtered_df.groupby("Sex Breakdown")["Numerator"].sum().reset_index()

donut = px.pie(
    donut_data,
    names="Sex Breakdown",
    values="Numerator",
    hole=0.5,
    title="Numerator Distribution by Sex"
)

st.plotly_chart(donut, use_container_width=True)

# -------------------------------------------------
# FUNNEL CHART (Numerator Trend by Year)
# -------------------------------------------------
funnel_data = filtered_df.groupby("Year_numeric")["Numerator"].sum().reset_index()

funnel = go.Figure(go.Funnel(
    y=funnel_data["Year_numeric"],
    x=funnel_data["Numerator"]
))

funnel.update_layout(title="Numerator Funnel by Year")

st.plotly_chart(funnel, use_container_width=True)

# -------------------------------------------------
# AGGREGATED DATASET
# -------------------------------------------------
agg_df = filtered_df.groupby(
    ["Year_numeric", "Sex Breakdown"]
).agg(
    Avg_Indicator=("Indicator value", "mean"),
    Total_Numerator=("Numerator", "sum"),
    Total_Denominator=("Denominator", "sum")
).reset_index()

st.subheader("Aggregated Dataset")
st.dataframe(agg_df, use_container_width=True)

# -------------------------------------------------
# MACHINE LEARNING PREDICTION (YEAR TREND)
# -------------------------------------------------
st.subheader("AI Prediction Panel")

# Use Persons only for clean prediction
persons_df = df[df["Sex Breakdown"] == "Persons"]

yearly_avg = persons_df.groupby("Year_numeric")["Indicator value"].mean().reset_index()

if len(yearly_avg) > 2:
    X = yearly_avg[["Year_numeric"]]
    y = yearly_avg["Indicator value"]

    model = LinearRegression()
    model.fit(X, y)

    future_year = st.number_input("Enter Future Year", min_value=2014, max_value=2035, value=2026)

    prediction = model.predict([[future_year]])[0]

    st.success(f"Predicted Indicator Value for {future_year}: {round(prediction,2)}")
else:
    st.warning("Not enough data for prediction.")

# -------------------------------------------------
# LOGOUT
# -------------------------------------------------
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()
