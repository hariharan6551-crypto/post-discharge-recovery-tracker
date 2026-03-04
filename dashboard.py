import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Post Discharge Recovery Tracker", layout="wide")

st.title("Post Discharge Social Support & Recovery Tracker")
st.markdown("Predictive Analytics & Recovery Monitoring Dashboard")

# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():
    return pd.read_csv("rhs_ml_dataset.csv")

df = load_data()

if "Year" not in df.columns:
    df["Year"] = 2023

# ===============================
# SIDEBAR FILTER
# ===============================

st.sidebar.header("Filter Panel")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["Year"].unique()))
filtered_df = df[df["Year"] == selected_year]

# ===============================
# KPI SECTION
# ===============================

st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", len(filtered_df))

if "Readmitted" in filtered_df.columns:
    readmission_rate = filtered_df["Readmitted"].mean() * 100
    col2.metric("Readmission Rate (%)", f"{readmission_rate:.2f}")

if "Length_of_Stay" in filtered_df.columns:
    col3.metric("Avg Length of Stay", f"{filtered_df['Length_of_Stay'].mean():.1f} Days")

# ===============================
# ANIMATED BAR
# ===============================

if "Readmitted" in df.columns:
    trend = df.groupby("Year")["Readmitted"].mean().reset_index()
    fig_bar = px.bar(trend, x="Year", y="Readmitted",
                     animation_frame="Year",
                     range_y=[0,1],
                     title="Yearly Readmission Trend")
    st.plotly_chart(fig_bar, use_container_width=True)

# ===============================
# DONUT CHART
# ===============================

if "Readmitted" in filtered_df.columns:
    fig_donut = px.pie(filtered_df,
                       names="Readmitted",
                       hole=0.5,
                       title="Readmission Distribution")
    st.plotly_chart(fig_donut, use_container_width=True)

# ===============================
# FUNNEL CHART
# ===============================

if "Social_Support_Level" in filtered_df.columns:
    funnel_data = filtered_df["Social_Support_Level"].value_counts().reset_index()
    funnel_data.columns = ["Stage", "Count"]
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_data["Stage"],
        x=funnel_data["Count"]
    ))
    st.plotly_chart(fig_funnel, use_container_width=True)

# ===============================
# MACHINE LEARNING MODEL (AUTO TRAIN)
# ===============================

if all(col in df.columns for col in ["Age", "Length_of_Stay", "Social_Support_Level", "Readmitted"]):

    X = df[["Age", "Length_of_Stay", "Social_Support_Level"]]
    y = df["Readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    st.sidebar.success(f"Model Accuracy: {accuracy:.2f}")

    # ===============================
    # PREDICTIVE PANEL
    # ===============================

    st.subheader("Predictive Risk Panel")

    colA, colB, colC = st.columns(3)

    age = colA.number_input("Age", 18, 100, 50)
    stay = colB.number_input("Length of Stay (Days)", 1, 60, 5)
    support = colC.selectbox("Social Support Level", [0,1,2])

    if st.button("Run Prediction"):

        input_data = pd.DataFrame({
            "Age": [age],
            "Length_of_Stay": [stay],
            "Social_Support_Level": [support]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"High Readmission Risk - Probability: {probability:.2f}")
        else:
            st.success(f"Low Readmission Risk - Probability: {probability:.2f}")
