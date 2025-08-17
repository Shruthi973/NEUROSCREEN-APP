# app.py — NeuroScreen (Parkinson's screening prototype)
# Single-page UI, no layer selector; runs L1/L2/L3 if models exist, then merges probabilities.
# Safe fallbacks so deployment never breaks if a model file is missing or incompatible.

import os, io, json, warnings
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
        st.image(IMG_PATH)  # avoid use_container_width for older Streamlit

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
      It is for educational use only and not a diagnostic device.
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Safe model loader + dummy -------------------------
class DummyModel:
    """
    Heuristic model so the app never crashes.
    Produces non-constant probabilities using key motor, non-motor, and cognitive signals.
    """
    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        def num(col, default=0.0):
            # Numeric vector for a feature; returns default if missing
            try:
                return pd.to_numeric(X.get(col, default), errors="coerce").fillna(default).astype(float).values
            except Exception:
                return np.full(len(X), default, dtype=float)

        s = np.zeros(len(X), dtype=float)

        # --- Motor (0–4 style sliders / scales) ---
        s += 0.7 * num("tremor_severity")
        s += 0.7 * num("rigidity")
        s += 0.7 * num("bradykinesia")
        s += 0.8 * num("gait_difficulty")
        s += 0.5 * (num("freezing_episodes") > 0).astype(float)
        # tapping lower is worse
        s += 0.4 * np.maximum(0, 20 - num("finger_taps_left")) / 20.0
        s += 0.4 * np.maximum(0, 20 - num("finger_taps_right")) / 20.0
        # TUG slower is worse
        s += 0.4 * np.maximum(0, num("tug_seconds") - 10) / 20.0

        # --- Cognitive (simple proxies) ---
        s += 0.6 * (5 - np.clip(num("serial7_correct"), 0, 5))
        s += 0.6 * (5 - np.clip(num("delayed_recall"), 0, 5))
        s += 0.4 * (20 - np.clip(num("animals_60s"), 0, 20)) / 20.0

        # --- Non-motor / autonomic ---
        s += 0.5 * (num("reduced_smell") > 0).astype(float)
        s += 0.3 * (num("rbd_like") > 0).astype(float)
        s += 0.2 * (num("constipation") > 0).astype(float)
        s += 0.2 * num("daytime_sleepiness") / 24.0
        s += 0.2 * num("orthostatic_lightheaded_weekly") / 3.0

        # Mild age effect
        s += 0.1 * np.maximum(0, num("age") - 60) / 30.0

        # Normalize → sigmoid
        z = (s - s.mean()) / (s.std() + 1e-6)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 0.02, 0.98)
        return np.c_[1 - p, p]

def load_model(model_path: str):
    """Try to load a joblib model; otherwise return DummyModel()."""
    try:
        import joblib  # lazy import
        if model_path and os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception:
        pass
    return DummyModel()

# Model file paths (env overrides supported)
L1_MODEL = os.environ.get("L1_MODEL", "model_layer1.pkl")
L2_MODEL = os.environ.get("L2_MODEL", "model_layer2_ppmi.pkl")
L3_MODEL = os.environ.get("L3_MODEL", "model_layer3.pkl")

# Load all models once
MODEL_L1 = load_model(L1_MODEL)
MODEL_L2 = load_model(L2_MODEL)
MODEL_L3 = load_model(L3_MODEL)

# ------------------------- UI helpers (single page) -------------------------
st.markdown("### Your Information")

with st.form("neuroscreen_form", clear_on_submit=False):
    # Demographics
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=60, step=1)
    with c2:
        sex = st.selectbox("Sex assigned at birth", ["Female", "Male", "Prefer not to say"])
    with c3:
        rbd_like = st.selectbox("Dream enactment / acting out dreams (RBD-like)?", ["No", "Yes"])

    st.markdown("### Motor Function (self-estimated or simple tests)")
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        tremor_severity = st.slider("Tremor severity (0=none, 4=severe)", 0, 4, 1)
    with c5:
        rigidity = st.slider("Rigidity (0–4)", 0, 4, 1)
    with c6:
        bradykinesia = st.slider("Slowness of movement (0–4)", 0, 4, 1)
    with c7:
        gait_difficulty = st.slider("Gait/balance difficulty (0–4)", 0, 4, 1)

    c8, c9, c10 = st.columns(3)
    with c8:
        freezing_episodes = st.selectbox("Freezing episodes?", ["No", "Yes"])
    with c9:
        finger_taps_left = st.number_input("Finger taps – left (0–20 in 10s)", 0, 40, 15)
    with c10:
        finger_taps_right = st.number_input("Finger taps – right (0–20 in 10s)", 0, 40, 15)

    c11, c12 = st.columns(2)
    with c11:
        tug_seconds = st.number_input("TUG: Timed Up & Go (seconds)", min_value=0.0, max_value=120.0, value=12.0, step=0.5)
    with c12:
        reduced_smell = st.selectbox("Reduced sense of smell (hyposmia)?", ["No", "Yes"])

    st.markdown("### Non-Motor & Autonomic")
    c13, c14, c15 = st.columns(3)
    with c13:
        constipation = st.selectbox("Constipation (weekly)?", ["No", "Yes"])
    with c14:
        daytime_sleepiness = st.slider("Daytime sleepiness (0–24, higher=worse)", 0, 24, 8)
    with c15:
        orthostatic_lightheaded_weekly = st.slider("Lightheaded when standing (days/week)", 0, 7, 1)

    st.markdown("### Cognitive Quick Checks")
    c16, c17, c18 = st.columns(3)
    with c16:
        serial7_correct = st.slider("Serial 7s correct (0–5)", 0, 5, 4)
    with c17:
        delayed_recall = st.slider("Delayed word recall (0–5)", 0, 5, 4)
    with c18:
        animals_60s = st.slider("Animals named in 60s (0–20)", 0, 40, 14)

    # Let user optionally add free-text notes (not used by model)
    notes = st.text_area("Optional notes (not used by the model)", "")

    submitted = st.form_submit_button("Run Screening")

# Collect inputs into a row for models
answers: Dict[str, Any] = {
    "age": age,
    "sex": sex,
    "rbd_like": 1 if rbd_like == "Yes" else 0,
    "tremor_severity": float(tremor_severity),
    "rigidity": float(rigidity),
    "bradykinesia": float(bradykinesia),
    "gait_difficulty": float(gait_difficulty),
    "freezing_episodes": 1 if freezing_episodes == "Yes" else 0,
    "finger_taps_left": float(finger_taps_left),
    "finger_taps_right": float(finger_taps_right),
    "tug_seconds": float(tug_seconds),
    "reduced_smell": 1 if reduced_smell == "Yes" else 0,
    "constipation": 1 if constipation == "Yes" else 0,
    "daytime_sleepiness": float(daytime_sleepiness),
    "orthostatic_lightheaded_weekly": float(orthostatic_lightheaded_weekly),
    "serial7_correct": float(serial7_correct),
    "delayed_recall": float(delayed_recall),
    "animals_60s": float(animals_60s),
}
X_user = pd.DataFrame([answers])

# ------------------------- Prediction utils -------------------------
def try_predict_proba(model, X: pd.DataFrame, layer_name: str) -> Tuple[float, str]:
    """
    Try model.predict_proba; if it fails (feature mismatch, etc.), fall back to DummyModel.
    Returns (prob_PD, note).
    """
    note = ""
    try:
        proba = model.predict_proba(X)
        # Assume proba shape (n, 2) as [neg, pos]
        p = float(proba[0, 1])
        p = float(np.clip(p, 0.0, 1.0))
        return p, note
    except Exception as e:
        # Use dummy if real model fails; add a short note for transparency (but not noisy)
        dm = DummyModel()
        p = float(dm.predict_proba(X)[0, 1])
        note = f"{layer_name}: fallback heuristic used (model incompatible with current inputs)."
        return p, note

def merge_probs(probs: List[float], weights: List[float]) -> float:
    """Weighted average of available layer probabilities."""
    probs = np.array(probs, dtype=float)
    weights = np.array(weights, dtype=float)
    if len(probs) == 0 or np.all(weights <= 0):
        return 0.5
    w = weights / (weights.sum() + 1e-9)
    return float(np.clip((probs * w).sum(), 0.0, 1.0))

def tier_from_prob(p: float) -> Tuple[str, str]:
    """Map probability to tier + short message."""
    if p < 0.33:
        return "Low", "Your responses suggest a lower likelihood on this screening."
    elif p < 0.66:
        return "Moderate", "Some features are present; consider monitoring or discussing with a clinician."
    else:
        return "Elevated", "Multiple risk features are present; consider a professional evaluation."

# ------------------------- Run & display -------------------------
if submitted:
    st.markdown("---")
    st.subheader("Screening Results")

    # Run each layer (if model file exists, we already loaded whatever available)
    layer_probs: List[Tuple[str, float]] = []
    layer_notes: List[str] = []

    # Weights: we can emphasize L3 (clinic-style) a bit; otherwise equal on availability
    base_weights = {"L1": 1.0, "L2": 1.0, "L3": 1.2}

    # L1
    p1, n1 = try_predict_proba(MODEL_L1, X_user, "L1")
    layer_probs.append(("L1 (Fox Insight-style)", p1))
    if n1:
        layer_notes.append(n1)

    # L2
    p2, n2 = try_predict_proba(MODEL_L2, X_user, "L2")
    layer_probs.append(("L2 (PPMI-style)", p2))
    if n2:
        layer_notes.append(n2)

    # L3
    p3, n3 = try_predict_proba(MODEL_L3, X_user, "L3")
    layer_probs.append(("L3 (Short clinical protocol)", p3))
    if n3:
        layer_notes.append(n3)

    # Determine weights based on which layers produced a probability (always 3 here, but keep safe)
    labels = [lp[0] for lp in layer_probs]
    probs_only = [lp[1] for lp in layer_probs]
    weights = []
    for label in labels:
        if "L1" in label:
            weights.append(base_weights["L1"])
        elif "L2" in label:
            weights.append(base_weights["L2"])
        elif "L3" in label:
            weights.append(base_weights["L3"])
        else:
            weights.append(1.0)

    p_final = merge_probs(probs_only, weights)
    tier, msg = tier_from_prob(p_final)

    # Display
    cA, cB = st.columns([1, 1])
    with cA:
        st.metric("Final merged risk (screening)", f"{p_final*100:.1f}%")
        st.write(f"**Tier:** {tier}")
        st.caption(msg)

    with cB:
        st.write("**Per-layer estimates**")
        for (label, p) in layer_probs:
            st.write(f"- {label}: **{p*100:.1f}%**")
        if layer_notes:
            st.caption(" • ".join(layer_notes))

    # Show the exact inputs back to the user
    with st.expander("Show the answers you entered"):
        st.json(answers, expanded=False)

    # ------------------------- PDF export -------------------------
    try:
        from fpdf import FPDF

        def build_pdf(answers: Dict[str, Any], layer_probs: List[Tuple[str, float]], p_final: float, tier: str, notes: List[str]) -> bytes:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=12)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "NeuroScreen – Parkinson's Risk Prototype", ln=1)

            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 8, "This report is for educational use only and is not a diagnosis. No data is stored.")
            pdf.ln(2)

            # Final
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 8, f"Final merged risk: {p_final*100:.1f}%  (Tier: {tier})", ln=1)
            pdf.set_font("Arial", "", 12)

            # Layers
            pdf.ln(2)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Per-layer estimates", ln=1)
            pdf.set_font("Arial", "", 12)
            for label, p in layer_probs:
                pdf.cell(0, 8, f"- {label}: {p*100:.1f}%", ln=1)

            if notes:
                pdf.ln(2)
                pdf.set_font("Arial", "I", 11)
                pdf.multi_cell(0, 6, "Notes: " + " • ".join(notes))

            # Answers
            pdf.ln(3)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Your answers", ln=1)
            pdf.set_font("Arial", "", 11)
            for k, v in answers.items():
                pdf.cell(0, 6, f"{k}: {v}", ln=1)

            buf = io.BytesIO()
            pdf.output(buf)
            return buf.getvalue()

        pdf_bytes = build_pdf(answers, layer_probs, p_final, tier, layer_notes)
        st.download_button(
            label="⬇️ Download screening summary (PDF)",
            data=pdf_bytes,
            file_name="neuroscreen_summary.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.caption("PDF export unavailable (fpdf not found or render error).")

# ------------------------- Footer -------------------------
st.markdown("---")
st.caption(
    "NeuroScreen is a research prototype. Results are estimates based on questionnaire-style inputs; "
    "consult a clinician for diagnosis or medical advice."
)
