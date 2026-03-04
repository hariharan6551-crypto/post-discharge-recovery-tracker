import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Post-Discharge Social Support and Recovery Tracker",
    layout="wide"
)

# -------------------------------------------------
# LOGIN SYSTEM
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
# TITLE
# -------------------------------------------------
st.title("Post-Discharge Social Support and Recovery Tracker")

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
df = pd.read_csv("data.csv")  # Make sure data.csv exists

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

year_filter = st.sidebar.multiselect(
    "Select Year",
    options=df["Year"].unique(),
    default=df["Year"].unique()
)

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

filtered_df = df[
    (df["Year"].isin(year_filter)) &
    (df["Gender"].isin(gender_filter))
]

# -------------------------------------------------
# AGGREGATED DATASET
# -------------------------------------------------
agg_df = filtered_df.groupby(["Year", "Gender"]).agg({
    "Treatment_Cost": "sum",
    "Patient_ID": "count"
}).reset_index()

agg_df.rename(columns={"Patient_ID": "Total_Patients"}, inplace=True)

# -------------------------------------------------
# KPI METRICS
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Patients", filtered_df.shape[0])
col2.metric("Total Treatment Cost", f"${filtered_df['Treatment_Cost'].sum():,.0f}")
col3.metric("Average Age", f"{filtered_df['Age'].mean():.1f}")

st.markdown("---")

# -------------------------------------------------
# BAR CHART
# -------------------------------------------------
bar_chart = px.bar(
    filtered_df,
    x="Department",
    title="Patients by Department",
)

st.plotly_chart(bar_chart, use_container_width=True)

# -------------------------------------------------
# DONUT CHART
# -------------------------------------------------
donut_chart = px.pie(
    filtered_df,
    names="Payment_Status",
    hole=0.5,
    title="Payment Status Distribution"
)

st.plotly_chart(donut_chart, use_container_width=True)

# -------------------------------------------------
# FUNNEL CHART
# -------------------------------------------------
funnel_data = filtered_df["Payment_Status"].value_counts().reset_index()
funnel_data.columns = ["Stage", "Count"]

funnel_chart = go.Figure(go.Funnel(
    y=funnel_data["Stage"],
    x=funnel_data["Count"]
))

funnel_chart.update_layout(title="Payment Funnel")

st.plotly_chart(funnel_chart, use_container_width=True)

# -------------------------------------------------
# MACHINE LEARNING PROTOTYPE MODEL
# -------------------------------------------------
st.markdown("## AI Prediction Panel")

# Aggregate daily patient count
daily_data = filtered_df.groupby("Date").size().reset_index(name="Patients")

if len(daily_data) > 1:
    daily_data["Date_Ordinal"] = daily_data["Date"].map(pd.Timestamp.toordinal)

    X = daily_data[["Date_Ordinal"]]
    y = daily_data["Patients"]

    model = LinearRegression()
    model.fit(X, y)

    future_date = st.date_input("Select Future Date for Prediction")

    future_ordinal = pd.Timestamp(future_date).toordinal()
    prediction = model.predict([[future_ordinal]])[0]

    st.success(f"Predicted Patients on {future_date}: {int(prediction)}")

else:
    st.warning("Not enough data for prediction.")

# -------------------------------------------------
# SHOW AGGREGATED DATASET
# -------------------------------------------------
st.markdown("## Aggregated Dataset")
st.dataframe(agg_df)

# -------------------------------------------------
# LOGOUT BUTTON
# -------------------------------------------------
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()
