import streamlit as st
import pandas as pd
import joblib
from catboost import Pool
import plotly.express as px

# =========================
# Severity Mapping + Colors
# =========================
severity_mapping = {1: "Low Severity 🚙", 2: "Medium Severity 🚗", 3: "High Severity 🚨"}
severity_colors = {1: "#2ecc71", 2: "#f1c40f", 3: "#e74c3c"}

# =========================
# Load Model & Metadata
# =========================
model = joblib.load("catboost_model.pkl")
cat_features = joblib.load("cat_features.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="Collision Severity Dashboard", layout="wide")
st.title("🚗 Collision Severity Prediction Dashboard")
st.markdown("Predict **legacy_collision_severity** using CatBoost with interactive dashboard style")

# =========================
# Custom CSS for Dashboard Style
# =========================
st.markdown("""
<style>
.section {
    background-color:#f0f2f6; 
    padding:15px; 
    border-radius:12px; 
    margin-bottom:15px;
    color: #000000;  /* نص أسود */
}
.box {
    background-color:#3498db; 
    padding:10px; 
    border-radius:10px; 
    margin-bottom:10px;
    color:white;      /* نص أبيض */
    font-weight:bold;
}
.big-font { 
    font-size:22px !important; 
    font-weight:bold; 
    color:white; 
    text-align:center; 
    padding:10px; 
    border-radius:12px; 
}
.stButton>button {
    background-color:#3498db; 
    color:white; 
    height:50px; 
    font-size:18px; 
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# User Inputs
# =========================
input_data = {}

# ===== Dropdown Options =====
severity_options = ["Low Severity 🚙", "Medium Severity 🚗", "High Severity 🚨"]
severity_mapping_reverse = {"Low Severity 🚙": 1, "Medium Severity 🚗": 2, "High Severity 🚨": 3}

police_options = ["No", "Yes"]
police_mapping_reverse = {"No": 0, "Yes": 1}

highway_options = ["Highway A", "Highway B", "Highway C"]
highway_mapping_reverse = {"Highway A": 0, "Highway B": 1, "Highway C": 2}

district_options = ["District A", "District B", "District C"]
district_mapping_reverse = {"District A": 0, "District B": 1, "District C": 2}

workday_options = ["Yes", "No"]
workday_mapping_reverse = {"No": 0, "Yes": 1}
weekend_mapping_reverse = {"No": 0, "Yes": 1}

# ===== Location Info Section =====
with st.container():
    st.markdown("<div class='section'><b>📍 Location Information</b></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown("<div class='box'>District</div>", unsafe_allow_html=True)
            selected_district = st.selectbox(" ", district_options)
            input_data["local_authority_ons_district"] = district_mapping_reverse[selected_district]
    with col2:
        with st.container():
            st.markdown("<div class='box'>Highway</div>", unsafe_allow_html=True)
            selected_highway = st.selectbox(" ", highway_options)
            input_data["local_authority_highway"] = highway_mapping_reverse[selected_highway]

# ===== Collision Details Section =====
with st.container():
    st.markdown("<div class='section'><b>🚨 Collision Details</b></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown("<div class='box'>Enhanced Severity</div>", unsafe_allow_html=True)
            selected_severity = st.selectbox(" ", severity_options)
            input_data["enhanced_severity_collision"] = severity_mapping_reverse[selected_severity]
    with col2:
        with st.container():
            st.markdown("<div class='box'>Police Officer Attended?</div>", unsafe_allow_html=True)
            selected_police = st.selectbox(" ", police_options)
            input_data["did_police_officer_attend_scene_of_collision"] = police_mapping_reverse[selected_police]

# ===== Time Info Section =====
with st.container():
    st.markdown("<div class='section'><b>⏰ Time Information</b></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container():
            st.markdown("<div class='box'>Day of Month</div>", unsafe_allow_html=True)
            input_data["day_month"] = st.number_input(" ", min_value=1, max_value=31, value=9)
    with col2:
        with st.container():
            st.markdown("<div class='box'>Hour</div>", unsafe_allow_html=True)
            hour_val = st.number_input(" ", min_value=0, max_value=23, value=12)
    with col3:
        with st.container():
            st.markdown("<div class='box'>Minute</div>", unsafe_allow_html=True)
            minute_val = st.number_input(" ", min_value=0, max_value=59, value=0)
    input_data["hour"] = hour_val + minute_val / 60

# ===== Workday & Weekend Section =====
with st.container():
    st.markdown("<div class='section'><b>📅 Day Type</b></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown("<div class='box'>Workday</div>", unsafe_allow_html=True)
            selected_workday = st.selectbox(" ", workday_options)
    with col2:
        if selected_workday == "Yes":
            selected_weekend = "No"
        else:
            selected_weekend = "Yes"
        st.markdown(f"<div class='box'>Weekend automatically set: {selected_weekend}</div>", unsafe_allow_html=True)

    input_data["Workday"] = workday_mapping_reverse[selected_workday]
    input_data["Weekend"] = weekend_mapping_reverse[selected_weekend]

# ===== باقي الأعمدة categorical / numeric =====
handled_features = [
    "day_month", "enhanced_severity_collision", "did_police_officer_attend_scene_of_collision",
    "local_authority_highway", "local_authority_ons_district", "Workday", "Weekend", "hour"
]

for feature in feature_names:
    if feature in handled_features or feature == "time_minutes":
        continue
    elif feature in cat_features:
        input_data[feature] = st.text_input(feature)
    else:
        input_data[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])

# ===== ترتيب الأعمدة كما في النموذج =====
model_columns = [f for f in feature_names if f != "time_minutes"]
input_df = input_df[model_columns]

# =========================
# Prediction Button
# =========================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("✅ Predict Severity"):
    predict_pool = Pool(input_df, cat_features=cat_features)
    prediction = model.predict(predict_pool)
    proba = model.predict_proba(predict_pool)

    predicted_class = prediction[0][0]
    predicted_label = severity_mapping.get(predicted_class, predicted_class)
    color = severity_colors.get(predicted_class, "#3498db")

    # ===== Result Box =====
    st.markdown(f"""
        <div class='big-font' style='background-color:{color};'>
            Predicted Severity: {predicted_label}
        </div>
        """, unsafe_allow_html=True)

    # ===== Probabilities Bar Chart =====
    st.subheader("📊 Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Severity": [severity_mapping.get(c, c) for c in model.classes_],
        "Probability (%)": (proba[0] * 100).round(2)
    })
    st.bar_chart(prob_df.set_index("Severity"))
    st.dataframe(prob_df)

    # ===== Pie Chart =====
    fig = px.pie(prob_df, names='Severity', values='Probability (%)',
                 color='Severity', color_discrete_map=severity_colors)
    st.plotly_chart(fig)
