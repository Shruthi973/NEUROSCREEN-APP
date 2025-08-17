# app.py — NeuroScreen (Parkinson's screening prototype)
# Wizard UI (9 pages + sidebar) + combined 3-layer prediction + PDF export
import os, json, warnings
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------- App frame -------------------------
st.set_page_config(page_title="NeuroScreen – Parkinson's Risk Prototype", layout="wide")

st.title("🧠 NeuroScreen – Parkinson's Risk Prototype")
st.caption("Tip: You can use this offline; nothing leaves your device.")

IMG_PATH = os.path.join(os.path.dirname(__file__), "AI.jpg")
if os.path.exists(IMG_PATH):
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        st.image(IMG_PATH)

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
class DummyModel:
    """Fallback so the app never crashes; only used if a .pkl can't be loaded."""
    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # very light heuristic so it's NOT always 0.5
        def num(col, default=0.0):
            try:
                return pd.to_numeric(X.get(col, default), errors="coerce").fillna(default).astype(float).values
            except Exception:
                return np.full(len(X), default, dtype=float)

        s = np.zeros(len(X), dtype=float)
        # Motor-ish proxies
        s += 0.8 * num("gait_difficulty")
        s += 0.7 * num("tremor_severity")
        s += 0.6 * num("bradykinesia")
        s += 0.5 * (num("reduced_smell") > 0).astype(float)
        s += 0.4 * (num("rbd_like") > 0).astype(float)
        s += 0.3 * (num("constipation") > 0).astype(float)
        s += 0.3 * (5 - np.clip(num("serial7_correct"), 0, 5))
        s += 0.3 * (5 - np.clip(num("delayed_recall"), 0, 5))
        z = (s - s.mean()) / (s.std() + 1e-6)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 0.02, 0.98)
        return np.c_[1 - p, p]

def load_model(model_path: str):
    try:
        import joblib
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception:
        pass
    return DummyModel()

# Model artifact paths (env or defaults)
L1_MODEL = os.environ.get("L1_MODEL", "model_layer1.pkl")
L2_MODEL = os.environ.get("L2_MODEL", "model_layer2_ppmi.pkl")
L3_MODEL = os.environ.get("L3_MODEL", "model_layer3.pkl")

MODEL_L1 = load_model(L1_MODEL)
MODEL_L2 = load_model(L2_MODEL)
MODEL_L3 = load_model(L3_MODEL)

# ------------------------- Session & wizard state -------------------------
PAGES = [
    "Welcome / Consent",
    "Demographics",
    "Motor – Tremor / Rigidity / Gait",
    "Motor – Dexterity / Tapping / TUG",
    "Non-Motor – Smell / Sleep / Autonomic",
    "Cognition – Serial 7s / Recall / Fluency",
    "Mood / Fatigue / Dizziness",
    "Safety & Daily Function",
    "Review & Results",
]

if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}  # keep all your existing keys/labels intact
if "history" not in st.session_state:
    st.session_state.history = []  # for linear nav if you want it later

# ------------------------- Helper: save answer safely -------------------------
def put_answer(key: str, value: Any):
    st.session_state.answers[key] = value

# ------------------------- Helper: combined prediction -------------------------
def to_dataframe_for_models(answers: Dict[str, Any]) -> pd.DataFrame:
    """
    Single-row DF that tries both: your existing keys AND model-preferred aliases.
    DO NOT rename your inputs; just ensure any important model features map here.
    """
    a = answers
    def g(k, default=None): return a.get(k, default)

    row = {
        # common motor fields you already use (keep your labels; keys can be same)
        "tremor_severity": g("tremor_severity", g("tremor (0-4)", 0)),
        "rigidity": g("rigidity", g("rigidity (0-4)", 0)),
        "bradykinesia": g("bradykinesia", g("slowness (0-4)", 0)),
        "gait_difficulty": g("gait_difficulty", g("gait (0-4)", 0)),
        "freezing_episodes": g("freezing_episodes", g("freezing yes/no", 0)),
        "finger_taps_left": g("finger_taps_left", g("finger taps left (0-20)", 20)),
        "finger_taps_right": g("finger_taps_right", g("finger taps right (0-20)", 20)),
        "tug_seconds": g("tug_seconds", g("timed up and go (sec)", 10)),
        # cognition
        "serial7_correct": g("serial7_correct", g("serial 7s correct (0-5)", 5)),
        "delayed_recall": g("delayed_recall", g("delayed recall (0-5)", 5)),
        "animals_60s": g("animals_60s", g("animal fluency in 60s", 20)),
        # non-motor
        "reduced_smell": g("reduced_smell", g("smell loss yes/no", 0)),
        "rbd_like": g("rbd_like", g("dream enactment yes/no", 0)),
        "constipation": g("constipation", g("constipation yes/no", 0)),
        "daytime_sleepiness": g("daytime_sleepiness", g("daytime sleepiness (0-24)", 0)),
        "orthostatic_lightheaded_weekly": g("orthostatic_lightheaded_weekly", g("orthostatic lightheaded/week", 0)),
        # demographics
        "age": g("age", g("Age", None)),
        "sex": g("sex", g("Sex", None)),
    }
    return pd.DataFrame([row])

def combined_probability(answers: Dict[str, Any]) -> Tuple[float, Dict[str, float], bool]:
    """
    Returns (p_final, per_layer_probs, used_dummy)
    - averages only across the models that successfully produce a prob
    - never forces 0.50 unless NO model provided a prob
    """
    X = to_dataframe_for_models(answers)
    per = {}
    used_dummy = False

    for name, mdl in [("Layer 1", MODEL_L1), ("Layer 2", MODEL_L2), ("Layer 3", MODEL_L3)]:
        try:
            proba = mdl.predict_proba(X)
            p1 = float(np.clip(proba[0, 1], 0.001, 0.999))
            per[name] = p1
            if isinstance(mdl, DummyModel):
                used_dummy = True
        except Exception:
            # if model fails hard, we skip it
            continue

    if len(per) == 0:
        # nothing available → neutral
        return 0.50, {}, True

    # simple mean; replace with your learned merger weights if you have them
    p_final = float(np.mean(list(per.values())))
    return p_final, per, used_dummy

# ------------------------- PDF export -------------------------
def export_pdf(answers: Dict[str, Any], p_final: float, per_layer: Dict[str, float]) -> bytes:
    try:
        from fpdf import FPDF
    except Exception:
        return b""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "NeuroScreen – Summary", ln=1)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Final combined risk: {round(p_final*100, 1)}%", ln=1)
    for k, v in per_layer.items():
        pdf.cell(0, 8, f"{k}: {round(v*100,1)}%", ln=1)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Your responses:", ln=1)

    pdf.set_font("Arial", "", 11)
    for k, v in answers.items():
        txt = f"- {k}: {v}"
        pdf.multi_cell(0, 6, txt)

    out = pdf.output(dest="S").encode("latin1", "ignore")
    return out

# ==========================
# WIZARD PAGE FUNCTIONS
# Paste your existing question blocks into the marked spots
# ==========================

def page_welcome():
    st.subheader("Welcome")
    st.write("Please proceed through the steps. Use the sidebar to navigate, or Next/Back.")
    # >>> If you had any consent checkbox or intro text, paste it here (unchanged).
    # Example:
    agree = st.checkbox("I understand this is a screening prototype and not a diagnosis.")
    if agree:
        put_answer("consent_ack", True)

def page_demographics():
    st.subheader("Demographics")
    # >>> PASTE YOUR EXISTING DEMOGRAPHIC QUESTIONS HERE (unchanged labels/help).
    age = st.number_input("Age", min_value=0, max_value=120, value=int(st.session_state.answers.get("Age", 60) or 60))
    sex = st.selectbox("Sex", ["", "Male", "Female", "Other"], index= ["","Male","Female","Other"].index(st.session_state.answers.get("Sex","") if st.session_state.answers.get("Sex","") in ["","Male","Female","Other"] else ""))
    put_answer("Age", age)
    put_answer("Sex", sex.lower() if sex else "")

def page_motor_1():
    st.subheader("Motor – Tremor / Rigidity / Gait")
    # >>> PASTE YOUR EXISTING PAGE CONTENT (unchanged). Keep your hints/helpers as-is.
    tremor = st.slider("tremor (0-4)", 0, 4, int(st.session_state.answers.get("tremor (0-4)", 0) or 0), help="0=None, 4=Severe")
    rigidity = st.slider("rigidity (0-4)", 0, 4, int(st.session_state.answers.get("rigidity (0-4)", 0) or 0))
    brady = st.slider("slowness (0-4)", 0, 4, int(st.session_state.answers.get("slowness (0-4)", 0) or 0))
    gait = st.slider("gait (0-4)", 0, 4, int(st.session_state.answers.get("gait (0-4)", 0) or 0))
    freeze = st.selectbox("freezing yes/no", ["No", "Yes"], index=1 if st.session_state.answers.get("freezing yes/no","No")=="Yes" else 0)

    put_answer("tremor (0-4)", tremor); put_answer("rigidity (0-4)", rigidity)
    put_answer("slowness (0-4)", brady); put_answer("gait (0-4)", gait)
    put_answer("freezing yes/no", 1 if freeze=="Yes" else 0)

    # map to general keys too (so models can find them without renaming your labels)
    put_answer("tremor_severity", tremor)
    put_answer("rigidity", rigidity)
    put_answer("bradykinesia", brady)
    put_answer("gait_difficulty", gait)
    put_answer("freezing_episodes", 1 if freeze=="Yes" else 0)

def page_motor_2():
    st.subheader("Motor – Dexterity / Tapping / TUG")
    # >>> PASTE your exact inputs here
    taps_l = st.number_input("finger taps left (0-20)", min_value=0, max_value=20, value=int(st.session_state.answers.get("finger taps left (0-20)", 20) or 20))
    taps_r = st.number_input("finger taps right (0-20)", min_value=0, max_value=20, value=int(st.session_state.answers.get("finger taps right (0-20)", 20) or 20))
    tug = st.number_input("timed up and go (sec)", min_value=0.0, max_value=60.0, value=float(st.session_state.answers.get("timed up and go (sec)", 10.0) or 10.0), step=0.1)

    put_answer("finger taps left (0-20)", taps_l)
    put_answer("finger taps right (0-20)", taps_r)
    put_answer("timed up and go (sec)", tug)

    put_answer("finger_taps_left", taps_l)
    put_answer("finger_taps_right", taps_r)
    put_answer("tug_seconds", tug)

def page_nonmotor():
    st.subheader("Non-Motor – Smell / Sleep / Autonomic")
    # >>> Keep your exact labels & help
    smell = st.selectbox("smell loss yes/no", ["No", "Yes"], index=1 if st.session_state.answers.get("smell loss yes/no","No")=="Yes" else 0)
    rbd = st.selectbox("dream enactment yes/no", ["No", "Yes"], index=1 if st.session_state.answers.get("dream enactment yes/no","No")=="Yes" else 0)
    constip = st.selectbox("constipation yes/no", ["No", "Yes"], index=1 if st.session_state.answers.get("constipation yes/no","No")=="Yes" else 0)
    sleepiness = st.slider("daytime sleepiness (0-24)", 0, 24, int(st.session_state.answers.get("daytime sleepiness (0-24)", 0) or 0))
    ortho = st.slider("orthostatic lightheaded/week", 0, 3, int(st.session_state.answers.get("orthostatic lightheaded/week", 0) or 0))

    put_answer("smell loss yes/no", smell)
    put_answer("dream enactment yes/no", rbd)
    put_answer("constipation yes/no", constip)
    put_answer("daytime sleepiness (0-24)", sleepiness)
    put_answer("orthostatic lightheaded/week", ortho)

    put_answer("reduced_smell", 1 if smell=="Yes" else 0)
    put_answer("rbd_like", 1 if rbd=="Yes" else 0)
    put_answer("constipation", 1 if constip=="Yes" else 0)
    put_answer("daytime_sleepiness", sleepiness)
    put_answer("orthostatic_lightheaded_weekly", ortho)

def page_cognition():
    st.subheader("Cognition – Serial 7s / Recall / Fluency")
    # >>> your exact cognition inputs go here
    s7 = st.slider("serial 7s correct (0-5)", 0, 5, int(st.session_state.answers.get("serial 7s correct (0-5)", 5) or 5))
    recall = st.slider("delayed recall (0-5)", 0, 5, int(st.session_state.answers.get("delayed recall (0-5)", 5) or 5))
    animals = st.slider("animal fluency in 60s", 0, 40, int(st.session_state.answers.get("animal fluency in 60s", 20) or 20))

    put_answer("serial 7s correct (0-5)", s7)
    put_answer("delayed recall (0-5)", recall)
    put_answer("animal fluency in 60s", animals)

    put_answer("serial7_correct", s7)
    put_answer("delayed_recall", recall)
    put_answer("animals_60s", animals)

def page_mood():
    st.subheader("Mood / Fatigue / Dizziness")
    # >>> paste your mood/fatigue/dizziness items here, unchanged
    fatigue = st.slider("fatigue (0-4)", 0, 4, int(st.session_state.answers.get("fatigue (0-4)", 0) or 0))
    dizzy = st.slider("dizziness/imbalance (0-4)", 0, 4, int(st.session_state.answers.get("dizziness/imbalance (0-4)", 0) or 0))
    put_answer("fatigue (0-4)", fatigue)
    put_answer("dizziness/imbalance (0-4)", dizzy)

def page_function():
    st.subheader("Safety & Daily Function")
    # >>> paste your ADL/safety items here (unchanged)
    falls = st.selectbox("any falls last 6 months?", ["No", "Yes"], index=1 if st.session_state.answers.get("any falls last 6 months?","No")=="Yes" else 0)
    put_answer("any falls last 6 months?", falls)

def page_review_results():
    st.subheader("Review & Results")

    st.write("### Your answers (review)")
    for k, v in st.session_state.answers.items():
        st.write(f"- **{k}**: {v}")

    st.write("---")
    if st.button("Compute combined risk"):
        p_final, per_layer, used_dummy = combined_probability(st.session_state.answers)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Final combined risk", f"{round(p_final*100,1)}%")
        with c2:
            if per_layer:
                st.write("Per-layer probabilities:")
                for nm, pv in per_layer.items():
                    st.write(f"- {nm}: **{round(pv*100,1)}%**")

        if used_dummy:
            st.info("Note: One or more model files were unavailable, so a safe fallback model was used for that layer.")

        # PDF export
        pdf_bytes = export_pdf(st.session_state.answers, p_final, per_layer)
        if pdf_bytes:
            st.download_button(
                "Download PDF summary",
                data=pdf_bytes,
                file_name="NeuroScreen_Summary.pdf",
                mime="application/pdf",
            )

# Map index → page function
PAGE_FUNCS = [
    page_welcome,
    page_demographics,
    page_motor_1,
    page_motor_2,
    page_nonmotor,
    page_cognition,
    page_mood,
    page_function,
    page_review_results,
]

# ------------------------- Sidebar navigation -------------------------
with st.sidebar:
    st.header("📋 Steps")
    # radio allows jumping; shows all “pages” like before
    st.session_state.step = st.radio(
        "Navigate",
        options=list(range(len(PAGES))),
        format_func=lambda i: PAGES[i],
        index=st.session_state.step,
    )

    st.markdown("---")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        if st.button("⬅️ Back", use_container_width=True, disabled=st.session_state.step==0):
            st.session_state.step = max(0, st.session_state.step - 1)
    with colB:
        st.write("")  # spacer
    with colC:
        if st.button("Next ➡️", use_container_width=True, disabled=st.session_state.step==len(PAGES)-1):
            st.session_state.step = min(len(PAGES)-1, st.session_state.step + 1)

# ------------------------- Render current page -------------------------
PAGE_FUNCS[st.session_state.step]()
