import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from datetime import datetime
import os

st.set_page_config(page_title="NeuroScreen", layout="wide")

# ✅ Load model and imputer
model_path = os.path.join(os.path.dirname(__file__), "model_rf_final.pkl")
imputer_path = os.path.join(os.path.dirname(__file__), "imputer.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    st.warning(f"⚠️ Model file not found at: {model_path}")

# ✅ Pages list
pages = {
    "Welcome": "🏠 Welcome",
    "Demographics": "👤 Demographics",
    "Motor": "🦵 Motor Symptoms",
    "Cognitive": "🧠 Cognitive",
    "MoodSleep": "😴 Mood & Sleep",
    "Function": "⚙️ Daily Functioning",
    "Predict": "📊 Submit & Predict",
    "Results": "✅ Results"
}

# ✅ Initialize page_nav only once
if "page_nav" not in st.session_state:
    st.session_state.page_nav = "Welcome"

# ✅ Sidebar Navigation (only updates if user explicitly clicks sidebar)
st.sidebar.title("🧭 Navigation")
if st.sidebar.button("Welcome"): st.session_state.page_nav = "Welcome"
if st.sidebar.button("Demographics"): st.session_state.page_nav = "Demographics"
if st.sidebar.button("Motor"): st.session_state.page_nav = "Motor"
if st.sidebar.button("Cognitive"): st.session_state.page_nav = "Cognitive"
if st.sidebar.button("Mood & Sleep"): st.session_state.page_nav = "MoodSleep"
if st.sidebar.button("Function"): st.session_state.page_nav = "Function"
if st.sidebar.button("Predict"): st.session_state.page_nav = "Predict"
if st.sidebar.button("Results"): st.session_state.page_nav = "Results"


# 🔒 HIPAA Compliance Notice
st.markdown("""
<div style='background-color:#e6f2ff; padding:10px; border-radius:8px; color:#000000'>
🔒 <strong>HIPAA Notice:</strong> NeuroScreen does not store or transmit your data. Your information stays on your device. Use this tool for educational purposes only.
</div>
""", unsafe_allow_html=True)

# ------------------ STEP 0: Welcome ------------------
if st.session_state.page_nav == "Welcome":
    st.title("🧠 NeuroScreen")
    st.subheader("🌟 Welcome to the Early Detection Tool")
    st.markdown("""
        NeuroScreen uses AI and behavioral data to estimate early Parkinson’s risk.  
        Click below to begin your personalized screening journey.
    """)
    if st.button("▶ Start Screening", key="start_btn"):
        st.session_state.page_nav = "Demographics"
        st.rerun()

# ------------------ STEP 1: Demographics ------------------
elif st.session_state.page_nav == "Demographics":
    st.header("Step 1: Demographics")
    st.session_state.age = st.number_input("Your Age", 18, 100)
    st.session_state.sex = st.selectbox("Sex", ["Male", "Female"])
    st.session_state.language = st.selectbox("Primary Language", ["English", "Spanish", "Other"])
    if st.button("Next →", key="demo_next"):
        st.session_state.page_nav = "Motor"
        st.rerun()

# ------------------ STEP 2: Motor Symptoms ------------------
elif st.session_state.page_nav == "Motor":
    st.header("Step 2: Motor Symptoms")
    st.session_state.tremor = st.slider("Tremor Severity", 0, 10)
    st.session_state.balance = st.radio("Balance Issues?", ["Yes", "No"])
    st.session_state.freeze = st.radio("Freezing Episodes?", ["Yes", "No"])
    if st.button("Next →", key="motor_next"):
        st.session_state.page_nav = "Cognitive"
        st.rerun()

# ------------------ STEP 3: Cognitive ------------------
elif st.session_state.page_nav == "Cognitive":
    st.header("Step 3: Cognitive Symptoms")
    st.session_state.memory = st.radio("Memory Issues?", ["Yes", "No"])
    st.session_state.task_switch = st.radio("Trouble Switching Tasks?", ["Yes", "No"])
    st.session_state.dual_task = st.radio("Difficulty with Dual Tasks (walk + talk)?", ["Yes", "No"])
    if st.button("Next →", key="cognitive_next"):
        st.session_state.page_nav = "MoodSleep"
        st.rerun()

# ------------------ STEP 4: Mood and Sleep ------------------
elif st.session_state.page_nav == "MoodSleep":
    st.header("Step 4: Mood & Sleep")
    st.session_state.anxiety = st.slider("Anxiety Level", 0, 10)
    st.session_state.depression = st.slider("Depression Level", 0, 10)
    st.session_state.sleep = st.radio("Frequent Sleep Issues?", ["Yes", "No"])
    if st.button("Next →", key="mood_next"):
        st.session_state.page_nav = "Function"
        st.rerun()

# ------------------ STEP 5: Daily Function ------------------
elif st.session_state.page_nav == "Function":
    st.header("Step 5: Daily Functioning")
    st.session_state.function = st.radio("Need help with daily activities?", ["Yes", "No"], key="function_radio")
    st.session_state.slow = st.radio("Feeling slowed down?", ["Yes", "No"], key="slow_radio")
    if st.button("Next →", key="function_next"):
        st.session_state.page_nav = "Predict"
        st.rerun()

# ------------------ STEP 6: Prediction ------------------
elif st.session_state.page_nav == "Predict":
    st.header("Step 6: Prediction Summary")
    st.markdown("🔍 Please complete the fields below (follow the instructions in brackets):")

    input_dict = {}

    # Demographics
    st.subheader("👤 Demographics")
    cols = st.columns(3)
    input_dict["screening_age"] = cols[0].number_input("screening_age (Enter your current age)", min_value=18, max_value=100)
    input_dict["sex"] = cols[1].selectbox("sex (1=Male, 2=Female)", [1, 2])
    input_dict["screening_language"] = cols[2].selectbox("screening_language (1=English, 2=Other)", [1, 2])

    # Psychological
    st.subheader("🧠 Psychological")
    input_dict["a_persistent_anxiety_total"] = st.slider("a_persistent_anxiety_total (Rate your anxiety over the past week, 0=none to 20=severe)", 0, 20)
    input_dict["promis_cat_v10_anxiety_4_complete"] = st.selectbox("Have you experienced nervousness or worry daily this week? (1=Yes, 0=No)", [1, 0])
    input_dict["promis_cat_v10_depression_4_complete"] = st.selectbox("Have you felt down, depressed, or hopeless this week? (1=Yes, 0=No)", [1, 0])
    input_dict["promis_cat_v10_sleep_disturbance_4_complete"] = st.selectbox("Have you had trouble sleeping most nights this week? (1=Yes, 0=No)", [1, 0])

    # Cognitive
    st.subheader("🧠 Cognitive (Memory & Recall)")
    st.markdown("Try this: Read and memorize the following words: _dog, noodles, king, store, brother_. Then wait 2 minutes.")
    for i in range(1, 6):
        input_dict[f"moca_firstrialmemory___{i}"] = st.selectbox(f"Did you remember word {i} immediately after hearing? (0=No, 1=Yes)", [0, 1])
    for i in range(1, 6):
        input_dict[f"moca_secondtrialmemory___{i}"] = st.selectbox(f"Did you recall word {i} after 2 minutes? (0=No, 1=Yes)", [0, 1])

    # Motor & Function
    st.subheader("🦵 Motor & Function")
    input_dict["irbd_quest1"] = st.number_input("irbd_quest1 (Did you have vivid dreams or move a lot in sleep? Rate from 0=no to 1=frequent)", 0.0, 1.0, step=0.05)
    input_dict["irbd_questionnaire_complete"] = st.selectbox("Completed the above dream question? (1=Yes, 0=No)", [1, 0])
    input_dict["freezing_of_gait_score_complete"] = st.selectbox("Did you feel stuck/freeze while walking this week? (1=Yes, 0=No)", [1, 0])
    input_dict["backwards_gait_complete"] = st.selectbox("Try walking backward 5 steps — were you steady? (1=Yes, 0=No)", [1, 0])
    input_dict["read_caterpillar_passage"] = st.selectbox("Read a short paragraph aloud clearly — could you do it? (1=Yes, 0=No)", [1.0, 0.0])
    input_dict["mdsupdrs_3_rigidity_sum"] = st.slider("mdsupdrs_3_rigidity_sum (Do limbs feel stiff when moving? 0=None to 20=Very rigid)", 0, 20)
    input_dict["mdsupdrs_3_tremor_sum"] = st.slider("mdsupdrs_3_tremor_sum (Tremor severity — 0=None to 20=Severe)", 0, 20)
    input_dict["tremor_score"] = st.number_input("tremor_score (Rate overall tremor from 0 to 5)", 0.0, 5.0, step=0.1)
    input_dict["phenotype_ration_tremor_pi"] = st.number_input("phenotype_ration_tremor_pi (If known — skip if unsure)", 0.0, 5.0, step=0.1)
    input_dict["mbt_dynamic_gait_subscale"] = st.slider("mbt_dynamic_gait_subscale (Walk 10 meters and turn — 0=easy, 20=very difficult)", 0, 20)

    # Balance & Attention
    st.subheader("🧍 Balance & Attention")
    input_dict["reactive_balance_data_and_notes_complete"] = st.selectbox("Stand and ask someone to gently nudge you — did you stay balanced? (1=Yes, 0=No)", [1, 0])
    input_dict["nasa_task_load_index_reactive_balance_complete"] = st.selectbox("While doing above balance task, were you mentally strained? (1=Yes, 0=No)", [1, 0])

    # Run prediction
    try:
        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[0][1]
        st.session_state.prob = round(prob * 100, 2)

        if st.button("🧪 Get My Results", key="get_results_btn"):
            st.session_state.page_nav = "Results"
            st.rerun()

    except Exception as e:
        st.error("⚠️ Prediction failed.")
        st.exception(e)

# ------------------ STEP 7: Results ------------------
elif st.session_state.page_nav == "Results":
    st.header("✅ Your Parkinson’s Risk Estimate")
    score = st.session_state.get("prob", 0.0)
    st.metric("Predicted Parkinson’s Risk", f"{score}%")

    if score >= 70:
        st.error("🔴 High Risk – Please consult a neurologist immediately.")
        advice = "You may require diagnostic testing or medical attention soon."
    elif score >= 40:
        st.warning("🟠 Moderate Risk – Consider follow-up.")
        advice = "Stay alert and speak with a doctor if symptoms worsen."
    else:
        st.success("🟢 Low Risk – No immediate concern.")
        advice = "You're in a healthy range. Remain aware and continue to monitor symptoms."

    st.markdown("🧠 _Feature importance (via SHAP) will be added in future versions._")

    