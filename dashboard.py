# =========================
# SIDEBAR FILTER
# =========================

st.sidebar.header("Filter Panel")

# Ensure Gender column exists
if "Gender" not in df.columns:
    st.error("Dataset missing required column: Gender")
    st.stop()

# Year Filter
selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Year"].unique())
)

# Gender Filter
selected_gender = st.sidebar.selectbox(
    "Select Gender",
    ["All"] + sorted(df["Gender"].unique())
)

# Apply Filters
filtered_df = df[df["Year"] == selected_year]

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
