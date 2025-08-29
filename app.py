# app.py  — NeuroScreen multi-layer PD risk tool (8 pages, no dropdowns)
# Run:  streamlit run app.py

import os, json, math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# --------------------- App config ---------------------
st.set_page_config(
    page_title="NeuroScreen",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File locations (adjust if your folders differ)
MODEL_L1 = "layer1_fox_out/model_layer1.pkl"
MODEL_L2 = "layer2_ppmi_out/model_layer2_ppmi.pkl"
MODEL_L3 = "layer3_shortened_out/model_layer3_shortened.pkl"

# Ensemble weights & thresholds
W = {"layer1": 0.2, "layer2": 0.4, "layer3": 0.4}
LOW_THR, HIGH_THR = 0.33, 0.66

# --------------------- Helpers (UI) -------------------
def likert_with_skip(label: str, key: str, help_text: Optional[str] = None) -> Optional[int]:
    c1, c2 = st.columns([1, 3])
    with c1:
        skip = st.checkbox("Prefer not to answer", key=f"{key}__skip")
    if skip:
        st.caption("↳ Skipped (blank for the model)")
        st.session_state[key] = None
        return None
    val = st.slider(label, min_value=0, max_value=4, value=0, key=key, help=help_text)
    return int(val)

def yesno_with_skip(label: str, key: str, help_text: Optional[str] = None) -> Optional[int]:
    choice = st.radio(
        label, ["Yes", "No", "Prefer not to answer"], key=key, horizontal=True, help=help_text
    )
    if choice == "Yes":
        return 1
    if choice == "No":
        return 0
    return None  # blank

def sex_radio(key: str) -> Optional[int]:
    choice = st.radio(
        "Sex", ["Male", "Female", "Prefer not to answer"], key=key, horizontal=True
    )
    if choice == "Male":
        return 1
    if choice == "Female":
        return 0
    return None

def number_with_skip(label: str, key: str, placeholder: str = "") -> Optional[float]:
    # use text_input so blank stays blank
    c1, c2 = st.columns([1, 3])
    with c1:
        skip = st.checkbox("Prefer not to answer", key=f"{key}__skip")
    if skip:
        st.caption("↳ Skipped (blank for the model)")
        st.session_state[key] = None
        return None
    txt = st.text_input(label, key=key, placeholder=placeholder)
    txt = txt.strip()
    if txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        st.warning("Please enter a number (or check 'Prefer not to answer').")
        return None

def section_scale_hint():
    st.caption("Scale: 0=None, 1=Slight, 2=Mild, 3=Moderate, 4=Severe. Use 'Prefer not to answer' if unsure.")

# --------------------- Helpers (models) ----------------
def unwrap_model(obj):
    # support dict bundles: {"pipeline":..., "features":[...], ...}
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"], obj.get("features")
    return obj, None

def expected_features(model, features_from_bundle: Optional[List[str]]) -> Optional[List[str]]:
    if features_from_bundle:
        return [str(c) for c in features_from_bundle]
    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]
    return None

def make_single_row_df(answers: Dict[str, Any], feature_list: List[str]) -> pd.DataFrame:
    row = {}
    for f in feature_list:
        v = answers.get(f, None)
        row[f] = (np.nan if v is None else v)
    return pd.DataFrame([row], columns=feature_list)

def predict_layer(model_path: str, answers: Dict[str, Any]) -> Tuple[Optional[float], Optional[List[str]]]:
    p = Path(model_path)
    if not p.exists():
        return None, None
    bundle = joblib.load(p)
    model, feats_hint = unwrap_model(bundle)
    feats = expected_features(model, feats_hint)
    # If model didn't record features, use whatever we have (unsafe, but rare)
    if not feats:
        feats = list(answers.keys())
    X = make_single_row_df(answers, feats).apply(pd.to_numeric, errors="coerce")
    try:
        prob = float(model.predict_proba(X)[:, 1][0])
        return max(0.0, min(1.0, prob)), feats
    except Exception as e:
        st.error(f"Layer at {model_path} failed: {e}")
        return None, feats

def weighted_blend(p1: Optional[float], p2: Optional[float], p3: Optional[float]) -> Tuple[float, str]:
    probs = {"layer1": p1, "layer2": p2, "layer3": p3}
    present = {k: v for k, v in probs.items() if v is not None and not math.isnan(v)}
    if not present:
        return float("nan"), "none"
    w = {k: W[k] for k in present.keys()}
    ws = sum(w.values())
    w = {k: v / ws for k, v in w.items()}  # renormalize
    p_final = sum(w[k] * present[k] for k in present.keys())
    used = "|".join(sorted(present.keys()))
    return float(p_final), used

def categorize(p_final: float) -> Tuple[str, str]:
    if not np.isfinite(p_final):
        return "unknown", "Not enough information to compute risk. Please answer a few more items."
    if p_final < LOW_THR:
        return "low", "Low risk: healthy habits, regular exercise, and routine check-ups are advisable."
    if p_final < HIGH_THR:
        return "medium", "Moderate risk: consider a neurological evaluation if symptoms persist or worsen."
    return "high", "High risk: please seek a clinical evaluation from a neurologist specializing in movement disorders."

# --------------------- Session state -------------------
if "page" not in st.session_state:
    st.session_state.page = 0
if "agree" not in st.session_state:
    st.session_state.agree = False
if "answers" not in st.session_state:
    st.session_state.answers = {}  # will hold all feature values

PAGES = [
    "Consent",
    "Demographics",
    "Smell & Bowel",
    "Urinary & Saliva",
    "Tremor & Balance",
    "Fatigue & Dizziness",
    "Memory & Anxiety",
    "Review & Results",
]

def goto(i):
    st.session_state.page = int(np.clip(i, 0, len(PAGES) - 1))

# --------------------- Sidebar ------------------------
st.sidebar.title("NeuroScreen")
st.sidebar.subheader("Navigate")
which = st.sidebar.radio("Pages", PAGES, index=st.session_state.page)
goto(PAGES.index(which))

# --------------------- Header -------------------------
from pathlib import Path
from PIL import Image
import streamlit as st

# point to your local PIC.png (same folder as app.py)
LOCAL_HEADER = Path("PIC.png")

if LOCAL_HEADER.exists():
    img = Image.open(LOCAL_HEADER)
    st.image(img, caption="PIC", use_column_width=True)  # works in all versions
else:
    st.warning("Local header image not found: PIC.png")


st.markdown(f"### HIPAA Notice & Consent")

st.write(
    "HIPAA Privacy Notice: This tool collects health-related answers for the limited purpose of risk triage and research. "
    "Data are stored only for the current session unless you choose to export. Do not enter personally identifiable information. "
    "By checking 'I Agree', you consent to use of your answers for the risk estimate shown in this app and to aggregate anonymized analytics."
)

st.session_state.agree = st.checkbox("I Agree", value=st.session_state.agree)

# Breadcrumb
crumbs = " > ".join(PAGES)
st.markdown(f"**Progress:** {crumbs}")

# Convenience
A = st.session_state.answers

# --------------------- Pages --------------------------
page = st.session_state.page

def nav_buttons():
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Back", use_container_width=True):
            goto(page - 1)
    with c2:
        if st.button("Next", disabled=(not st.session_state.agree and page == 0), use_container_width=True):
            goto(page + 1)

# Page 0: Consent
if page == 0:
    st.header("Consent")
    st.info("Please confirm consent to proceed. You can still browse, but 'Compute risk' requires consent.")
    nav_buttons()

# Page 1: Demographics
elif page == 1:
    st.header("Demographics")
    age = number_with_skip("Age (years)", key="age_num")
    sex_val = sex_radio("sex_radio")

    # Map to all layers
    A["age"] = age
    A["AGE_AT_VISIT"] = age
    A["screening_age"] = age
    A["Sex"] = sex_val
    A["SEX"] = sex_val
    A["sex"] = sex_val

    nav_buttons()

# Page 2: Smell & Bowel
elif page == 2:
    st.header("Smell & Bowel")
    section_scale_hint()

    # L1 Likert
    A["ImpactMoveHyposmia"] = likert_with_skip(
        "Reduced sense of smell — severity", "ImpactMoveHyposmia",
        "Sniff coffee grounds or soap; rate how strong it smells."
    )
    A["ImpactMoveConstipation"] = likert_with_skip(
        "Constipation affecting daily life — severity", "ImpactMoveConstipation",
        "Past 2 weeks: hard stools, straining, <3 BMs/week."
    )
    # L2 flag
    A["UPSIT_PRESENT"] = yesno_with_skip(
        "Reduced smell present (Yes/No)?", "UPSIT_PRESENT",
        "If coffee/soap/lotion smells faint or absent, choose Yes."
    )
    nav_buttons()

# Page 3: Urinary & Saliva
elif page == 3:
    st.header("Urinary & Saliva")
    section_scale_hint()

    A["ImpactMoveUrinary"] = likert_with_skip(
        "Urinary urgency/frequency — severity", "ImpactMoveUrinary",
        "Do you rush to the bathroom or go more often than usual?"
    )
    A["MoveSaliva"] = likert_with_skip(
        "Excess saliva/drooling — severity", "MoveSaliva",
        "Need to swallow often; drooling on pillow?"
    )
    A["URIN_PRESENT"] = yesno_with_skip(
        "Urinary problems present (Yes/No)?", "URIN_PRESENT",
        "Urgency, frequency, or nighttime bathroom trips >2."
    )
    A["NP1URIN_OL"] = likert_with_skip(
        "Urinary difficulties (overall) — severity", "NP1URIN_OL"
    )
    A["PQUEST_SOURCE_OL"] = yesno_with_skip(
        "Did you complete these questions yourself? (Yes/No)", "PQUEST_SOURCE_OL"
    )
    nav_buttons()

# Page 4: Tremor & Balance
elif page == 4:
    st.header("Tremor & Balance")
    section_scale_hint()

    A["ImpactMoveTremor"] = likert_with_skip(
        "Tremor affecting tasks — severity", "ImpactMoveTremor",
        "Hold your hands out 10 sec; try lifting a cup; rate impact."
    )
    A["ARMLGSHK_OL"] = likert_with_skip("Arm/leg shaking (tremor) — severity", "ARMLGSHK_OL")
    A["SHUFFLE_OL"] = likert_with_skip("Shuffling steps — severity", "SHUFFLE_OL")
    A["MVSLOW_OL"] = likert_with_skip("Slowness of movement — severity", "MVSLOW_OL")
    A["POORBAL_OL"] = likert_with_skip("Poor balance / near-falls — severity", "POORBAL_OL")
    A["FTSTUCK_OL"] = likert_with_skip("Feet feel stuck / freezing — severity", "FTSTUCK_OL")

    st.subheader("Walking ability (short test)")
    st.caption("If safe, try 10 meters. If you use a cane/walker, select that option. Skip if unsafe.")
    walk_choice = st.radio(
        "Can you walk across a room unaided?", ["Yes", "With aid", "No", "Prefer not to answer"],
        horizontal=True, key="screening_walk_radio"
    )
    if walk_choice == "Yes":
        A["screening_walk"] = 1.0
    elif walk_choice == "With aid":
        A["screening_walk"] = 0.5
    elif walk_choice == "No":
        A["screening_walk"] = 0.0
    else:
        A["screening_walk"] = None

    A["dt_tug_time"] = number_with_skip(
        "Timed Up & Go (seconds)", key="dt_tug_time",
        placeholder="e.g., 9.6"
    )

    nav_buttons()

# Page 5: Fatigue & Dizziness
elif page == 5:
    st.header("Fatigue & Dizziness")
    section_scale_hint()

    A["ImpactThinkBodyFatigue"] = likert_with_skip(
        "Fatigue slowing thinking/activity — severity", "ImpactThinkBodyFatigue",
        "Is daytime energy low; do tasks feel slower?"
    )
    A["ImpactThinkDizzy"] = likert_with_skip(
        "Dizziness / light-headedness — severity", "ImpactThinkDizzy",
        "Stand up and count to 10—did you feel woozy?"
    )
    A["VOICSFTR_OL"] = likert_with_skip("Softer / quieter voice — severity", "VOICSFTR_OL")
    A["LSSXPRSS_OL"] = likert_with_skip("Reduced facial expression — severity", "LSSXPRSS_OL")

    A["RBD_PRESENT"] = yesno_with_skip(
        "Acting out dreams during sleep (kicking/shouting)?", "RBD_PRESENT",
        "Ask a bed partner if unsure."
    )
    if A.get("RBD_PRESENT", None) == 1:
        src = st.radio(
            "Who noticed/diagnosed this?",
            ["Self (1)", "Bed partner (2)", "Physician (3)", "Prefer not to answer"],
            horizontal=True, key="RBD_SOURCE_OL_radio"
        )
        if src == "Self (1)":
            A["RBD_SOURCE_OL"] = 1
        elif src == "Bed partner (2)":
            A["RBD_SOURCE_OL"] = 2
        elif src == "Physician (3)":
            A["RBD_SOURCE_OL"] = 3
        else:
            A["RBD_SOURCE_OL"] = None
    else:
        A["RBD_SOURCE_OL"] = None

    nav_buttons()

# Page 6: Memory & Anxiety
elif page == 6:
    st.header("Memory & Anxiety")
    section_scale_hint()

    A["ImpactThinkMemory"] = likert_with_skip(
        "Memory / word-finding issues — severity", "ImpactThinkMemory",
        "Names/appointments/words harder than usual?"
    )
    A["Anxiety"] = likert_with_skip(
        "Feeling anxious / worried — severity (past 2 weeks)", "Anxiety"
    )

    st.subheader("Quick cognitive checks")
    A["DIFFRECALL"] = likert_with_skip(
        "Word recall difficulty — severity", "DIFFRECALL",
        "Say 3 words (e.g., apple, chair, penny); after 1 minute, try to recall them."
    )
    A["MCAVIGIL"] = likert_with_skip(
        "Sustained attention issues — severity", "MCAVIGIL",
        "Count backward by 7s for 30 sec or focus on a timer; rate difficulty."
    )
    A["MEM_PRESENT"] = yesno_with_skip(
        "Memory problems present (Yes/No)?", "MEM_PRESENT",
        "Misplacing items, repeating questions, forgetting recent events."
    )

    st.subheader("Short protocol items")
    # 0–10 self ratings
    def zero_ten_with_skip(label, key):
        c1, c2 = st.columns([1, 3])
        with c1:
            skip = st.checkbox("Prefer not to answer", key=f"{key}__skip")
        if skip:
            st.caption("↳ Skipped (blank for the model)")
            A[key] = None
            return
        v = st.slider(label, 0, 10, 0, key=key)
        A[key] = int(v)

    zero_ten_with_skip("Anxiety score (0–10)", "anxiety_score")
    zero_ten_with_skip("Conscious movement score (0–10)", "conscious_movement_score")

    A["cdte"] = yesno_with_skip(
        "Clock-drawing done today?", "cdte",
        "Draw an analog clock showing 10 past 11; choose Yes if performed now."
    )
    A["cogdt"] = yesno_with_skip(
        "Orientation item done today (date & location)?", "cogdt"
    )

    # L2 motor/usability items
    st.subheader("Fine motor / function")
    A["TRBBUTTN_OL"] = likert_with_skip("Buttons/clasps difficult — severity", "TRBBUTTN_OL")
    A["TRBUPCHR_OL"] = likert_with_skip("Trouble using phone/computer — severity", "TRBUPCHR_OL")
    A["WRTSMLR_OL"] = likert_with_skip("Handwriting smaller — severity", "WRTSMLR_OL")
    A["DFCLTYTYPE_OL"] = likert_with_skip("Difficulty typing — severity", "DFCLTYTYPE_OL")

    nav_buttons()

# Page 7: Review & Results
else:
    st.header("Review & Results")

    # Build final answers row, including ID and mapped age/sex already
    # Ensure an ID for the row:
    if "id" not in A or not A.get("id"):
        A["id"] = "A001"

    # Save a one-row CSV for your CLI flow
    cols_order = [
        "id",
        # L1
        "ImpactMoveHyposmia","ImpactMoveConstipation","Anxiety","ImpactMoveUrinary",
        "ImpactMoveTremor","ImpactThinkBodyFatigue","ImpactThinkDizzy","ImpactMoveImbalance",
        "ImpactThinkMemory","MoveSaliva","age","Sex",
        # L2
        "AGE_AT_VISIT","SEX","DIFFRECALL","MCAVIGIL","NP1URIN_OL","RBD_SOURCE_OL","UPSIT_PRESENT",
        "RBD_PRESENT","URIN_PRESENT","MEM_PRESENT","PQUEST_SOURCE_OL","TRBUPCHR_OL","WRTSMLR_OL",
        "DFCLTYTYPE_OL","VOICSFTR_OL","FTSTUCK_OL","LSSXPRSS_OL","ARMLGSHK_OL","TRBBUTTN_OL",
        "SHUFFLE_OL","MVSLOW_OL","POORBAL_OL",
        # L3
        "screening_age","sex","anxiety_score","conscious_movement_score","screening_walk",
        "cdte","cogdt","dt_tug_time"
    ]
    # Ensure all keys exist
    for k in cols_order:
        A.setdefault(k, None)

    df_out = pd.DataFrame([[A.get(c, None) for c in cols_order]], columns=cols_order)
    df_out.to_csv("app_inputs.csv", index=False)
    st.caption("Saved current answers → app_inputs.csv")

    # Predict per layer
    p1, f1 = predict_layer(MODEL_L1, A)
    p2, f2 = predict_layer(MODEL_L2, A)
    p3, f3 = predict_layer(MODEL_L3, A)

    used_feats = {
        "layer1": f1 if f1 else [],
        "layer2": f2 if f2 else [],
        "layer3": f3 if f3 else [],
    }

    p_final, used_layers = weighted_blend(p1, p2, p3)
    risk_cat, advice = categorize(p_final)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Layer 1 prob", f"{'—' if p1 is None else f'{p1*100:.1f}%'}")
    with c2:
        st.metric("Layer 2 prob", f"{'—' if p2 is None else f'{p2*100:.1f}%'}")
    with c3:
        st.metric("Layer 3 prob", f"{'—' if p3 is None else f'{p3*100:.1f}%'}")

    st.subheader("Final PD risk")
    if np.isfinite(p_final):
        st.success(f"**{p_final*100:.1f}%**  →  **{risk_cat.upper()}**")
    else:
        st.error("Not enough information to compute a final risk.")

    st.write(advice)

    st.caption(f"Used layers: {used_layers or 'none'} (weights: L1={W['layer1']}, L2={W['layer2']}, L3={W['layer3']})")

    st.divider()
    with st.expander("Show raw feature row (debug)"):
        st.dataframe(df_out)

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Back", use_container_width=True):
            goto(page - 1)
    with c2:
        st.button("Finish", use_container_width=True, disabled=not st.session_state.agree)

