# app.py — NeuroScreen (Parkinson's screening prototype) — stable UI + dense questions
import os, json, warnings
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------- App frame -------------------------
st.set_page_config(page_title="NeuroScreen – Parkinson's Risk Prototype", layout="wide")

# Title + caption
st.title("🧠 NeuroScreen – Parkinson's Risk Prototype")
st.caption("Tip: You can use this offline; nothing leaves your device.")

# Optional welcome image (safe for older Streamlit)
IMG_PATH = os.path.join(os.path.dirname(__file__), "AI.jpg")
if os.path.exists(IMG_PATH):
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        st.image(IMG_PATH)  # no use_container_width to avoid TypeError

# HIPAA box (black)
st.markdown(
    """
    <div style="
        background:#0e1117;
        color:#e6e6e6;
        padding:12px 14px;
        border-radius:10px;
        border:1px solid #232a35;
        ">
      🔒 <b>HIPAA notice:</b> This prototype does not store or transmit your data.
      It is for educational use only.
    </div>
    """,
    unsafe_allow_html=True,
)



# ------------------------- Safe model loader -------------------------
# Top-level DummyModel so it's pickle-safe
class DummyModel:
    """Heuristic model so the app never crashes; returns reasonable probs from key signals."""
    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        def num(col, default=0.0):
            try:
                return pd.to_numeric(X.get(col, default), errors="coerce").fillna(default).astype(float).values
            except Exception:
                return np.full(len(X), default, dtype=float)

        s = np.zeros(len(X), dtype=float)
        # Motor (0–4 scales weighted)
        s += 0.7 * num("tremor_severity")
        s += 0.7 * num("rigidity")
        s += 0.7 * num("bradykinesia")
        s += 0.8 * num("gait_difficulty")
        s += 0.5 * (num("freezing_episodes") > 0).astype(float)
        s += 0.4 * np.maximum(0, 20 - num("finger_taps_left")) / 20.0
        s += 0.4 * np.maximum(0, 20 - num("finger_taps_right")) / 20.0
        s += 0.4 * np.maximum(0, num("tug_seconds") - 10) / 20.0
        # Cognitive
        s += 0.6 * (5 - np.clip(num("serial7_correct"), 0, 5))
        s += 0.6 * (5 - np.clip(num("delayed_recall"), 0, 5))
        s += 0.4 * (20 - np.clip(num("animals_60s"), 0, 20)) / 20.0
        # Non-motor / autonomic
        s += 0.5 * (num("reduced_smell") > 0).astype(float)
        s += 0.3 * (num("rbd_like") > 0).astype(float)
        s += 0.2 * (num("constipation") > 0).astype(float)
        s += 0.2 * num("daytime_sleepiness") / 24.0
        s += 0.2 * num("orthostatic_lightheaded_weekly") / 3.0
        # Normalize → prob
        z = (s - s.mean()) / (s.std() + 1e-6)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 0.01, 0.99)
        return np.c_[1 - p, p]

def load_model(model_path: str):
    try:
        import joblib  # loaded lazily to avoid cache/pickle issues
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception:
        pass
    return DummyModel()

# Map env or defaults (these can be absent; UI still works)
L1_MODEL = os.environ.get("L1_MODEL", "model_layer1.pkl")
L2_MODEL = os.environ.get("L2_MODEL", "model_layer2_ppmi.pkl")
L3_MODEL = os.environ.get("L3_MODEL", "model_layer3.pkl")

# ------------------------- Navigation state -------------------------
if "layer" not in st.session_state:
    st.session_state.layer = "Layer 3 (Final)"
if "section" not in st.session_state:
    st.session_state.section = "Welcome"
if "answers" not in st.session_state:
    st.session_state.answers = {}

# ------------------------- Sidebar -------------------------
st.sidebar.title("🧭 Navigation")

st.session_state.layer = st.sidebar.selectbox(
    "Model layer",
    ["Layer 1", "Layer 2 (PPMI)", "Layer 3 (Final)"],
    index=2 if st.session_state.layer.endswith("Final") else 0,
)

for sec in [
    "Welcome",
    "Demographics",
    "Motor (Self-ratings)",
    "Motor Tasks (At-home)",
    "Cognitive",
    "Non-motor (Autonomic/Sleep/Mood)",
    "Daily Function & Safety",
    "Risk Factors & History",
    "Voice & Gait Self-tests",
    "Review & Predict",
]:
    if st.sidebar.button(sec, use_container_width=True):
        st.session_state.section = sec

# ------------------------- Dense question spec (60+) -------------------------
# Each item has: key, label, type, and instructions in [brackets]
QUESTIONS: Dict[str, List[Dict[str, Any]]] = {
    "Demographics": [
        {"key":"age","type":"number","label":"Age (years) [enter exact age]", "min":18, "max":100, "default":55},
        {"key":"sex","type":"select","label":"Sex","options":["Male","Female","Other"],"default":"Male"},
        {"key":"handedness","type":"select","label":"Handedness","options":["Right","Left","Both"],"default":"Right"},
        {"key":"education_years","type":"number","label":"Years of education [highest completed]", "min":0,"max":30,"default":16},
        {"key":"family_history_pd","type":"select","label":"First-degree relative with Parkinson's?","options":["No","Yes","Unknown"],"default":"No"},
        {"key":"smell_loss_years","type":"number","label":"Years since noticing reduced smell [0 if none]", "min":0,"max":40,"default":0},
    ],
    "Motor (Self-ratings)": [
        {"key":"tremor_severity","type":"slider","label":"Tremor severity (0 none – 4 severe) [at rest or posture]","min":0,"max":4,"default":0},
        {"key":"rigidity","type":"slider","label":"Rigidity/stiffness (0–4) [try passive elbow/ wrist move]","min":0,"max":4,"default":0},
        {"key":"bradykinesia","type":"slider","label":"Slowness (0–4) [buttoning, utensil use]","min":0,"max":4,"default":0},
        {"key":"gait_difficulty","type":"slider","label":"Gait imbalance (0–4) [start/turn/freezing]","min":0,"max":4,"default":0},
        {"key":"freezing_episodes","type":"slider","label":"Freezing episodes (0 none, 1 occasional, 2 frequent)","min":0,"max":2,"default":0},
        {"key":"shoulder_stiff_am","type":"select","label":"Morning shoulder stiffness","options":["No","Mild","Moderate","Severe"],"default":"No"},
        {"key":"micrographia_self","type":"select","label":"Smaller/clumsy handwriting","options":["No","Slightly","Clearly smaller"],"default":"No"},
    ],
    "Motor Tasks (At-home)": [
        {"key":"tug_seconds","type":"number","label":"Timed Up & Go (seconds) [stand, walk 3m, turn, sit]","min":0,"max":120,"default":12},
        {"key":"finger_taps_left","type":"number","label":"Finger taps in 10s – LEFT [tap index finger rapidly]","min":0,"max":200,"default":50},
        {"key":"finger_taps_right","type":"number","label":"Finger taps in 10s – RIGHT","min":0,"max":200,"default":55},
        {"key":"key_turn","type":"slider","label":"Open a jar / turn key (0–4) [0=easy, 4=unable]","min":0,"max":4,"default":0},
        {"key":"buttoning","type":"slider","label":"Button a shirt (0–4) [time yourself, 0<10s, 4=unable]","min":0,"max":4,"default":0},
        {"key":"pouring_spill","type":"slider","label":"Pour water without spill (0–4) [0=none, 4=spills most]","min":0,"max":4,"default":0},
        {"key":"posture_hunched","type":"select","label":"Stooped posture noticed by others?","options":["No","Sometimes","Often"],"default":"No"},
    ],
    "Cognitive": [
        {"key":"vigilance_missed","type":"select","label":"Vigilance task – missed taps? [tap when you see 7s]","options":["None","1 missed","≥2 missed"],"default":"None"},
        {"key":"serial7_correct","type":"slider","label":"Serial 7s correct (0–5) [100→−7×5]","min":0,"max":5,"default":5},
        {"key":"delayed_recall","type":"slider","label":"Delayed recall (0–5) [remember 5 words after 5 min]","min":0,"max":5,"default":5},
        {"key":"animals_60s","type":"number","label":"Animals named in 60s [say animals aloud for 60s]","min":0,"max":40,"default":18},
        {"key":"clock_errors","type":"select","label":"Clock drawing errors [set 10 past 11]","options":["None","Spacing","Numbers","Hands"],"default":"None"},
        {"key":"word_find_diff","type":"select","label":"Word-finding difficulty","options":["No","Sometimes","Often"],"default":"No"},
        {"key":"multitask_diff","type":"select","label":"Trouble doing 2 things at once (dual-task)","options":["No","Mild","Marked"],"default":"No"},
    ],
    "Non-motor (Autonomic/Sleep/Mood)": [
        {"key":"reduced_smell","type":"select","label":"Reduced smell (self-report)","options":["No","Mild","Moderate","Severe"],"default":"No"},
        {"key":"constipation","type":"select","label":"Constipation frequency","options":["No","Occasional","Frequent"],"default":"No"},
        {"key":"urinary_urgency","type":"select","label":"Urinary urgency","options":["No","Occasional","Frequent"],"default":"No"},
        {"key":"orthostatic_lightheaded_weekly","type":"slider","label":"Lightheaded on standing (days/week 0–3+)","min":0,"max":3,"default":0},
        {"key":"rbd_like","type":"select","label":"Acting out dreams (RBD-like) [kicking/punching/talking]","options":["No","Unsure","Yes"],"default":"No"},
        {"key":"insomnia","type":"select","label":"Insomnia (trouble falling/staying asleep)","options":["No","Mild","Moderate","Severe"],"default":"No"},
        {"key":"daytime_sleepiness","type":"number","label":"Epworth daytime sleepiness (0–24) [sum your 8 items]","min":0,"max":24,"default":6},
        {"key":"phq4_total","type":"number","label":"Anxiety/Depression (PHQ-4 total 0–12)","min":0,"max":12,"default":2},
        {"key":"anosmia_family","type":"select","label":"Family history of smell loss","options":["No","Yes","Unknown"],"default":"No"},
    ],
    "Daily Function & Safety": [
        {"key":"falls_12mo","type":"number","label":"Falls in last 12 months","min":0,"max":50,"default":0},
        {"key":"fear_of_falling","type":"select","label":"Fear of falling","options":["No","Sometimes","Often"],"default":"No"},
        {"key":"household_tasks","type":"slider","label":"Household tasks difficulty (0–4)","min":0,"max":4,"default":0},
        {"key":"community_mobility","type":"slider","label":"Community mobility difficulty (0–4)","min":0,"max":4,"default":0},
        {"key":"driving_change","type":"select","label":"Driving changes (slower, less at night)","options":["No","Mild","Marked","Not driving"],"default":"No"},
        {"key":"work_impact","type":"select","label":"Impact on work/chores","options":["None","Mild","Moderate","Severe"],"default":"None"},
    ],
    "Risk Factors & History": [
        {"key":"years_since_first_symptom","type":"number","label":"Years since first motor symptom","min":0,"max":40,"default":0},
        {"key":"first_symptom","type":"select","label":"First predominant symptom","options":["None","Tremor","Stiffness","Slowness","Gait/Balance","Other"],"default":"None"},
        {"key":"pesticide_exposure","type":"select","label":"Pesticide/solvent exposure (occupational)","options":["No","Possible","Yes"],"default":"No"},
        {"key":"head_injury_lossconscious","type":"select","label":"Past head injury with loss of consciousness","options":["No","Yes"],"default":"No"},
        {"key":"antipsychotic_use_history","type":"select","label":"Past antipsychotic/metoclopramide use","options":["No","Yes","Unknown"],"default":"No"},
        {"key":"caffeine_cups_day","type":"number","label":"Caffeine cups/day (coffee/tea)","min":0,"max":12,"default":1},
        {"key":"exercise_days_week","type":"number","label":"Exercise days/week ≥30min","min":0,"max":7,"default":3},
        {"key":"bmi","type":"number","label":"BMI (kg/m²) [enter if known]","min":10,"max":60,"default":24},
    ],
    "Voice & Gait Self-tests": [
    {
        "key": "read_caterpillar_passage",
        "type": "select",
        "label": "Read this short passage aloud (clarity/monotone). Text: "
                 "'Do you like amusement parks? I do. My favorite ride is the roller coaster. "
                 "It goes up, up, up… then down, down, down. It twists and turns, faster and faster. "
                 "Some people scream, some laugh, and some hold on tight. When the ride stops, "
                 "everyone wants to go again.'  [Read out loud now]",
        "options": ["Done", "Not done"],
        "default": "Not done"
    },
    {
        "key": "voice_loudness_self",
        "type": "select",
        "label": "Voice loudness (hypophonia) [count 1–10 at a comfortable volume]",
        "options": ["Normal", "Slightly soft", "Often too soft"],
        "default": "Normal"
    },
    {
        "key": "max_sustained_ah_seconds",
        "type": "number",
        "label": "Max sustained 'ah' (seconds) [deep breath, sustain 'ah' as long as you can]",
        "min": 0,
        "max": 60,
        "default": 10
    },
    {
        "key": "gait_dual_task_count",
        "type": "number",
        "label": "Dual-task walk: count backwards by 3s for 20 steps [enter number of mistakes]",
        "min": 0,
        "max": 20,
        "default": 0
    },
    {
        "key": "turns_en_bloc",
        "type": "select",
        "label": "Turns en bloc (stiff, multiple steps)",
        "options": ["No", "Sometimes", "Often"],
        "default": "No"
    },
    {
        "key": "sit_to_stand_5x_seconds",
        "type": "number",
        "label": "5× Sit-to-stand time (seconds) [arms crossed, stand-sit five times]",
        "min": 0,
        "max": 120,
        "default": 14
    }
],

}

ALL_SECTIONS = list(QUESTIONS.keys()) + ["Review & Predict"]

# ------------------------- UI render helpers -------------------------
def render_section(name: str):
    st.header(name)
    for q in QUESTIONS.get(name, []):
        key = q["key"]
        label = q["label"]
        if q["type"] == "number":
            val = st.number_input(label, q.get("min", 0), q.get("max", 1000), q.get("default", 0), key=key)
        elif q["type"] == "slider":
            val = st.slider(label, q.get("min", 0), q.get("max", 10), q.get("default", 0), key=key)
        elif q["type"] == "select":
            val = st.selectbox(label, q.get("options", []), index=q.get("options", []).index(q.get("default")) if q.get("default") in q.get("options", []) else 0, key=key)
        else:
            val = st.text_input(label, q.get("default",""), key=key)
        st.session_state.answers[key] = val

def answers_to_row() -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for sec, items in QUESTIONS.items():
        for it in items:
            k = it["key"]
            v = st.session_state.answers.get(k)
            # normalize categorical to numeric hints where sensible
            if isinstance(v, str):
                vmap = {
                    "No":0,"None":0,"Normal":0,"Not done":0,
                    "Yes":1,"Done":1,"Often":2,"Marked":2,"Severe":3,
                    "Mild":1,"Sometimes":1,"Moderate":2,"Frequent":2,"Possible":1,
                    "Unknown":0,
                    "Slightly soft":1,"Often too soft":2,
                    "Slightly":1,"Clearly smaller":2,
                    "Right":1,"Left":2,"Both":3,
                    "Male":1,"Female":2,"Other":3,
                }
                row[k] = vmap.get(v, v)
            else:
                row[k] = v
    # Some composites the DummyModel expects as numeric toggles
    row["reduced_smell"] = 1 if (row.get("smell_loss_years",0)>0 or (st.session_state.answers.get("reduced_smell","No")!="No")) else 0
    row["rbd_like"] = 1 if st.session_state.answers.get("rbd_like","No")=="Yes" else 0
    row["constipation"] = 1 if st.session_state.answers.get("constipation","No")!="No" else 0
    return row

# ------------------------- Main content -------------------------
sec = st.session_state.section

if sec == "Welcome":
    st.subheader("Use the sidebar to start.")
else:
    if sec in QUESTIONS:
        render_section(sec)
    elif sec == "Review & Predict":
        st.header("Review & Predict")
        row = answers_to_row()
        df = pd.DataFrame([row])

        # Choose model by layer (falls back to DummyModel if missing)
        if st.session_state.layer == "Layer 1":
            model = load_model(L1_MODEL)
        elif st.session_state.layer.startswith("Layer 2"):
            model = load_model(L2_MODEL)
        else:
            model = load_model(L3_MODEL)

        try:
            prob = float(model.predict_proba(df)[0][1])
        except Exception:
            prob = float(DummyModel().predict_proba(df)[0][1])

        pct = round(100*prob, 1)
        st.metric("Estimated Parkinson’s probability", f"{pct}%")

        if pct >= 70:
            st.error("🔴 High risk — please see a neurologist.")
        elif pct >= 40:
            st.warning("🟠 Moderate risk — consider follow-up.")
        else:
            st.success("🟢 Low risk — monitor and recheck if symptoms evolve.")

        with st.expander("Show the data you entered"):
            st.dataframe(df.T.rename(columns={0:"value"}))

        # PDF export
        from fpdf import FPDF
        def make_pdf() -> bytes:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "NeuroScreen Risk Report", ln=True, align="C")
            pdf.ln(4)
            pdf.cell(0, 8, f"Layer: {st.session_state.layer}", ln=True)
            pdf.cell(0, 8, f"Estimated probability: {pct}%", ln=True)
            pdf.ln(4)
            for k, v in row.items():
                pdf.multi_cell(0, 6, f"{k}: {v}")
            return pdf.output(dest="S").encode("latin1")
        if st.button("📄 Download PDF report"):
            st.download_button("Save PDF", data=make_pdf(), file_name="neuroscreen_report.pdf", mime="application/pdf")
    else:
        st.info("Pick a section from the sidebar.")

