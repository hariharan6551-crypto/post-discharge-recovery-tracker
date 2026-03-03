import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(layout="wide")
st.title("🚀 Post-Discharge Social Support and Recovery Tracker")

# 🔵 Animation - Loading Effect
with st.spinner("⏳ Loading Dataset & Training Model..."):
    time.sleep(1)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("rhs_ml_dataset.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ==============================
# CLEAN DATA
# ==============================

df = df.replace(r"[^\d.]", "", regex=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

df = df.fillna(0)

# ==============================
# DEFINE TARGET
# ==============================
target = df.columns[-1]

X = df.drop(columns=[target])
y = df[target]

X = pd.get_dummies(X)
X, y = X.align(y, join="inner", axis=0)

# ==============================
# TRAIN MODEL WITH PROGRESS BAR
# ==============================

st.subheader("🤖 Training Model...")

progress_bar = st.progress(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)

# Simulated progress animation
for percent in range(1, 101, 20):
    time.sleep(0.2)
    progress_bar.progress(percent)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.success("✅ Model Training Completed!")

st.metric("📉 MAE", round(mae, 3))
st.metric("📊 R² Score", round(r2, 3))

pickle.dump(model, open("model.pkl", "wb"))

# ==============================
# ANIMATED VISUALIZATION
# ==============================

st.subheader("📊 Animated Visualization")

with st.expander("📈 Target Distribution (Click to Expand)"):
    fig1 = px.histogram(df, x=target, title="Target Distribution")
    fig1.update_layout(transition_duration=500)
    st.plotly_chart(fig1, use_container_width=True)

with st.expander("🔥 Feature Importance"):
    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    fig2 = px.bar(
        feature_df,
        x="Feature",
        y="Importance",
        title="Feature Importance"
    )

    fig2.update_layout(transition_duration=800)

    st.plotly_chart(fig2, use_container_width=True)

# ==============================
# ANIMATED PREDICTION PANEL
# ==============================

st.subheader("🔮 Prediction Panel")

input_data = {}

cols = st.columns(2)

for i, col in enumerate(X.columns):
    with cols[i % 2]:
        input_data[col] = st.number_input(
            f"Enter {col}",
            value=0.0
        )

if st.button("🚀 Predict Now"):
    with st.spinner("🧠 AI Thinking..."):
        time.sleep(1)

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    st.balloons()  # Animation effect
    st.success(f"🎯 Predicted Output: {prediction[0]}")




