# app.py  — NeuroScreen multi-layer PD risk tool (8 pages, no dropdowns)
# Run:  streamlit run app.py

import os, json, math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---- PAGE CONFIG & IMPORTS (PASTE AT LINE 1) ----
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NeuroScreen",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- GLOBAL STYLE (PASTE RIGHT AFTER set_page_config) ----
HERO_CSS = """
<style>
.hero {
  padding: 36px 28px;
  border-radius: 20px;
  background: radial-gradient(1200px 600px at 10% -10%, #1f2a44 0%, rgba(31,42,68,0) 50%),
              linear-gradient(135deg,#1e1258 0%, #0e5bb3 50%, #06b6d4 100%);
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
  color: #fff;
}
.hero h1 { margin: 0 0 6px 0; font-size: 42px; letter-spacing: .3px;}
.hero p  { opacity: .95; font-size: 17px; margin: 0;}
.card {
  border-radius: 18px; padding: 18px; background: var(--secondary-background-color,#131A2A);
  border: 1px solid rgba(255,255,255,.06); box-shadow: 0 8px 24px rgba(0,0,0,.25);
}
</style>
"""
st.markdown(HERO_CSS, unsafe_allow_html=True)

# --------------------- App config ---------------------
# (Removed duplicate st.set_page_config — Streamlit allows it only once)

# File locations (adjust if your folders differ)
MODEL_L1 = "model_layer1.pkl"
MODEL_L2 = "model_layer2_ppmi.pkl"
MODEL_L3 = "model_layer3.pkl"

# Ensemble weights & thresholds
W = {"layer1": 0.2, "layer2": 0.4, "layer3": 0.4}
LOW_THR, HIGH_THR = 0.33, 0.66


# --------------------- Helpers (UI) -------------------

def likert_with_skip(
    label: str,
    key: str,
    desc: str = None,
    help_text: Optional[str] = None,
) -> Optional[int]:
    # 1) Question + slider
    val = st.slider(label, min_value=0, max_value=4, value=0, key=key, help=help_text)
    # 2) Gray helper line (if provided)
    if desc: st.caption(desc)
    # 3) Prefer-not checkbox BELOW the slider
    skip = st.checkbox("Prefer not to answer", key=f"{key}__skip")
    if skip:
        st.caption("↳ Skipped (blank for the model)")
        st.session_state[key] = None
        return None
    return int(val)


def yesno_with_skip(
    label: str,
    key: str,
    desc: str = None,
    help_text: Optional[str] = None,
) -> Optional[int]:
    choice = st.radio(
        label,
        ["Yes", "No", "Prefer not to answer"],
        key=key,
        horizontal=True,
        help=help_text
    )
    if desc: st.caption(desc)
    if choice == "Yes":
        return 1
    if choice == "No":
        return 0
    return None  # blank

def sex_radio(
    key: str,
    desc: str = None,
) -> Optional[int]:
    choice = st.radio(
        "Sex",
        ["Male", "Female", "Prefer not to answer"],
        key=key,
        horizontal=True
    )
    if desc: st.caption(desc)
    if choice == "Male":
        return 1
    if choice == "Female":
        return 0
    return None

def number_with_skip(
    label: str,
    key: str,
    placeholder: str = "",
    desc: str = None,
) -> Optional[float]:
    # 1) Question + text input
    txt = st.text_input(label, key=key, placeholder=placeholder)
    txt = txt.strip()
    # 2) Gray helper line (if provided)
    if desc: st.caption(desc)
    # 3) Prefer-not checkbox BELOW the input
    skip = st.checkbox("Prefer not to answer", key=f"{key}__skip")
    if skip:
        st.caption("↳ Skipped (blank for the model)")
        st.session_state[key] = None
        return None
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
    # ---- GUARDRAIL: exclude layer if ZERO usable inputs (per requirement) ----
    nn = sum(answers.get(f, None) is not None for f in feats)
    if nn == 0:
        return None, feats  # mark as "Insufficient data" upstream

    X = make_single_row_df(answers, feats).apply(pd.to_numeric, errors="coerce")
    try:
        prob = float(model.predict_proba(X)[:, 1][0])
        return max(0.0, min(1.0, prob)), feats
    except Exception:
        # Any prediction error => treat as missing for blending, but keep feats for diagnostics
        return None, feats

# ---------- NEW: Renormalize-only-present-layers blending (drop-in) ----------
def blend_probs(weights: Dict[str, float],
                probs: Dict[str, Optional[float]]
               ) -> Tuple[Optional[float], Dict[str, float], List[str]]:
    """
    weights: {"layer1": 0.2, "layer2": 0.4, "layer3": 0.4}
    probs:   {"layer1": None or 0.xx, "layer2": 0.yy, "layer3": 0.zz}
    returns: (p_final or None, renorm_weights dict, missing list)
    """
    present = {k: p for k, p in probs.items() if p is not None and np.isfinite(p)}
    missing = [k for k, p in probs.items() if p is None or not np.isfinite(p)]
    if not present:
        return None, {}, missing

    total_w = sum(weights[k] for k in present.keys())
    # Defensive: if total_w is 0 (shouldn't happen with your W), fall back to equal weights
    if total_w <= 0:
        eq = 1.0 / len(present)
        renorm = {k: eq for k in present.keys()}
    else:
        renorm = {k: weights[k] / total_w for k in present.keys()}

    p_final = sum(renorm[k] * present[k] for k in present.keys())
    return float(p_final), renorm, missing

def categorize(p_final: float) -> Tuple[str, str]:
    if p_final is None or not np.isfinite(p_final):
        return "unknown", "Not enough information to compute risk. Please answer a few more items."
    if p_final < LOW_THR:
        return "low", "Low risk: healthy habits, regular exercise, and routine check-ups are advisable."
    if p_final < HIGH_THR:
        return "medium", "Moderate risk: consider a neurological evaluation if symptoms persist or worsen."
    return "high", "High risk: please seek a clinical evaluation from a neurologist specializing in movement disorders."

# --- diagnostics helpers ---
def model_exists(path): 
    p = Path(path)
    return p.exists(), str(p.resolve())

def nonnull_count_for(features, answers):
    return sum(answers.get(f, None) is not None for f in (features or []))

def safe_predict(name, path, answers):
    ok, full = model_exists(path)
    if not ok:
        return None, None, f"{name}: model file missing at {full}"
    try:
        prob, feats = predict_layer(path, answers)
        # Note: predict_layer already enforces zero-input exclusion.
        if prob is None:
            nn = nonnull_count_for(feats, answers)
            if nn == 0:
                return None, feats, f"{name}: Insufficient data (0 usable inputs)"
            return None, feats, f"{name}: predict_proba failed or returned None (nonnull features fed: {nn})"
        return prob, feats, f"{name}: OK (nonnull features fed: {nonnull_count_for(feats, answers)})"
    except Exception as e:
        return None, None, f"{name}: exception -> {e}"

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
    i = int(np.clip(i, 0, len(PAGES) - 1))
    st.session_state.page = i
    # tell the next run to sync the sidebar radio BEFORE it's rendered
    st.session_state["nav_force_sync"] = True


# --------------------- Sidebar ------------------------
# Keep the sidebar radio in sync with page when navigation came from buttons
if st.session_state.get("nav_force_sync", False):
    st.session_state["nav_radio"] = PAGES[st.session_state.page]
    st.session_state["nav_force_sync"] = False

def _sync_page_from_radio():
    # When the user clicks the radio, update page
    st.session_state.page = PAGES.index(st.session_state["nav_radio"])

st.sidebar.title("NeuroScreen")
st.sidebar.subheader("Navigate")

st.sidebar.radio(
    "Pages",
    PAGES,
    key="nav_radio",
    index=st.session_state.page,
    on_change=_sync_page_from_radio,  # radio controls page
)


# --------------------- Header -------------------------
from pathlib import Path
from PIL import Image
import streamlit as st

# point to your local PIC.png (same folder as app.py)
LOCAL_HEADER = Path("PIC.png")

if LOCAL_HEADER.exists():
    img = Image.open(LOCAL_HEADER)
    st.image(img, use_column_width=True)  # works in all versions
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
    # ---- FORM WRAPPER ----
    with st.form("form0"):
        c1, c2 = st.columns([1,1])
        with c1:
            back = st.form_submit_button("Back", use_container_width=True)
        with c2:
            nextp = st.form_submit_button("Next", disabled=(not st.session_state.agree), use_container_width=True)
    if back:
        goto(page - 1)
    if nextp:
        goto(page + 1)

# Page 1: Demographics
elif page == 1:
    st.header("Demographics")
    # ---- FORM WRAPPER ----
    with st.form("form1"):
        age = number_with_skip(
            "Age (years)",
            key="age_num",
            desc="Enter your age in whole years."
        )
        sex_val = sex_radio(
            "sex_radio",
            desc="Select your sex."
        )

        # Map to all layers
        A["age"] = age
        A["AGE_AT_VISIT"] = age
        A["screening_age"] = age
        A["Sex"] = sex_val
        A["SEX"] = sex_val
        A["sex"] = sex_val

        c1, c2 = st.columns([1,1])
        with c1:
            back = st.form_submit_button("Back", use_container_width=True)
        with c2:
            nextp = st.form_submit_button("Next", use_container_width=True)
    if back:
        goto(page - 1)
    if nextp:
        goto(page + 1)

# Page 2: Smell & Bowel
elif page == 2:
    st.header("Smell & Bowel")
    section_scale_hint()

    # ---- FORM WRAPPER ----
    with st.form("form2"):
        # L1 Likert
        A["ImpactMoveHyposmia"] = likert_with_skip(
            "Reduced sense of smell — severity",
            "ImpactMoveHyposmia",
            desc="Sniff coffee grounds or soap and rate how strong the smell is."
        )
        A["ImpactMoveConstipation"] = likert_with_skip(
            "Constipation affecting daily life — severity",
            "ImpactMoveConstipation",
            desc="Past 2 weeks: hard stools, straining, or fewer than 3 bowel movements per week."
        )
        # L2 flag
        A["UPSIT_PRESENT"] = yesno_with_skip(
            "Reduced smell present (Yes/No)?",
            "UPSIT_PRESENT",
            desc="If coffee/soap/lotion smells faint or absent, choose ‘Yes’."
        )

        c1, c2 = st.columns([1,1])
        with c1:
            back = st.form_submit_button("Back", use_container_width=True)
        with c2:
            nextp = st.form_submit_button("Next", use_container_width=True)
    if back:
        goto(page - 1)
    if nextp:
        goto(page + 1)

# Page 3: Urinary & Saliva
elif page == 3:
    st.header("Urinary & Saliva")
    section_scale_hint()

    # ---- FORM WRAPPER ----
    with st.form("form3"):
        A["ImpactMoveUrinary"] = likert_with_skip(
            "Urinary urgency/frequency — severity",
            "ImpactMoveUrinary",
            desc="Do you rush to the bathroom or go more often than usual?"
        )
        A["MoveSaliva"] = likert_with_skip(
            "Excess saliva/drooling — severity",
            "MoveSaliva",
            desc="Do you need to swallow often or notice drool on your pillow?"
        )
        A["URIN_PRESENT"] = yesno_with_skip(
            "Urinary problems present (Yes/No)?",
            "URIN_PRESENT",
            desc="Urgency, frequency, or nighttime bathroom trips more than twice."
        )
        A["NP1URIN_OL"] = likert_with_skip(
            "Urinary difficulties (overall) — severity",
            "NP1URIN_OL",
            desc="Overall rating of urinary difficulties in the past 2 weeks."
        )
        A["PQUEST_SOURCE_OL"] = yesno_with_skip(
            "Did you complete these questions yourself? (Yes/No)",
            "PQUEST_SOURCE_OL",
            desc="Choose ‘Yes’ if you personally answered these questions today."
        )

        c1, c2 = st.columns([1,1])
        with c1:
            back = st.form_submit_button("Back", use_container_width=True)
        with c2:
            nextp = st.form_submit_button("Next", use_container_width=True)
    if back:
        goto(page - 1)
    if nextp:
        goto(page + 1)

# Page 4: Tremor & Balance
elif page == 4:
    st.header("Tremor & Balance")
    section_scale_hint()

    # ---- FORM WRAPPER ----
    with st.form("form4"):
        A["ImpactMoveTremor"] = likert_with_skip(
            "Tremor affecting tasks — severity",
            "ImpactMoveTremor",
            desc="Hold your hands out for 10 seconds or try lifting a cup; rate the impact."
        )
        A["ARMLGSHK_OL"] = likert_with_skip(
            "Arm/leg shaking (tremor) — severity",
            "ARMLGSHK_OL",
            desc="Shaking in the arms or legs at rest or during action; rate typical severity."
        )
        A["SHUFFLE_OL"] = likert_with_skip(
            "Shuffling steps — severity",
            "SHUFFLE_OL",
            desc="Short, shuffling steps when walking; rate how often/severe this feels."
        )
        A["MVSLOW_OL"] = likert_with_skip(
            "Slowness of movement — severity",
            "MVSLOW_OL",
            desc="General slowness (bradykinesia): smaller/fewer movements or slower actions."
        )
        A["POORBAL_OL"] = likert_with_skip(
            "Poor balance / near-falls — severity",
            "POORBAL_OL",
            desc="Unsteady balance, near-falls, or needing support to steady yourself."
        )
        A["FTSTUCK_OL"] = likert_with_skip(
            "Feet feel stuck / freezing — severity",
            "FTSTUCK_OL",
            desc="Feet ‘stick’ when starting, turning, or in narrow spaces; rate frequency/severity."
        )

        st.subheader("Walking ability (short test)")
        st.caption("If safe, try 10 meters. If you use a cane/walker, select that option. Skip if unsafe.")
        walk_choice = st.radio(
            "Can you walk across a room unaided?",
            ["Yes", "With aid", "No", "Prefer not to answer"],
            horizontal=True,
            key="screening_walk_radio"
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
            "Timed Up & Go (seconds)",
            key="dt_tug_time",
            placeholder="e.g., 9.6",
            desc="From seated: stand, walk 3 m, turn, return, and sit; enter the total seconds."
        )

        c1, c2 = st.columns([1,1])
        with c1:
            back = st.form_submit_button("Back", use_container_width=True)
        with c2:
            nextp = st.form_submit_button("Next", use_container_width=True)
    if back:
        goto(page - 1)
    if nextp:
        goto(page + 1)

# Page 5: Fatigue & Dizziness
elif page == 5:
    st.header("Fatigue & Dizziness")
    section_scale_hint()

    # ---- FORM WRAPPER ----
    with st.form("form5"):
        A["ImpactThinkBodyFatigue"] = likert_with_skip(
            "Fatigue slowing thinking/activity — severity",
            "ImpactThinkBodyFatigue",
            desc="Is daytime energy low; do tasks feel slower than usual?"
        )
        A["ImpactThinkDizzy"] = likert_with_skip(
            "Dizziness / light-headedness — severity",
            "ImpactThinkDizzy",
            desc="Stand up and count to 10—did you feel woozy or light-headed?"
        )
        A["VOICSFTR_OL"] = likert_with_skip("Softer / quieter voice — severity", "VOICSFTR_OL")
        A["LSSXPRSS_OL"] = likert_with_skip("Reduced facial expression — severity", "LSSXPRSS_OL")

        A["RBD_PRESENT"] = yesno_with_skip(
            "Acting out dreams during sleep (kicking/shouting)?",
            "RBD_PRESENT",
            desc="Ask a bed partner if unsure; choose ‘Yes’ if dream enactment is present."
        )
        if A.get("RBD_PRESENT", None) == 1:
            src = st.radio(
                "Who noticed/diagnosed this?",
                ["Self (1)", "Bed partner (2)", "Physician (3)", "Prefer not to answer"],
                horizontal=True,
                key="RBD_SOURCE_OL_radio"
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

        c1, c2 = st.columns([1,1])
        with c1:
            back = st.form_submit_button("Back", use_container_width=True)
        with c2:
            nextp = st.form_submit_button("Next", use_container_width=True)
    if back:
        goto(page - 1)
    if nextp:
        goto(page + 1)

# Page 6: Memory & Anxiety
elif page == 6:
    st.header("Memory & Anxiety")
    section_scale_hint()

    # ---- FORM WRAPPER ----
    with st.form("form6"):
        A["ImpactThinkMemory"] = likert_with_skip(
            "Memory / word-finding issues — severity",
            "ImpactThinkMemory",
            desc="Are names, appointments, or common words harder to recall than usual?"
        )
        A["Anxiety"] = likert_with_skip(
            "Feeling anxious / worried — severity (past 2 weeks)",
            "Anxiety",
            desc="Overall anxiety or worry level over the last two weeks."
        )

        st.subheader("Quick cognitive checks")
        A["DIFFRECALL"] = likert_with_skip(
            "Word recall difficulty — severity",
            "DIFFRECALL",
            desc="Say 3 words (e.g., apple, chair, penny); after 1 minute, try to recall them."
        )
        A["MCAVIGIL"] = likert_with_skip(
            "Sustained attention issues — severity",
            "MCAVIGIL",
            desc="Try counting backward by 7s for 30 seconds; rate difficulty focusing."
        )
        A["MEM_PRESENT"] = yesno_with_skip(
            "Memory problems present (Yes/No)?",
            "MEM_PRESENT",
            desc="Misplacing items, repeating questions, or forgetting recent events."
        )

        st.subheader("Short protocol items")
        # 0–10 self ratings
        def zero_ten_with_skip(label, key):
            c1, c2 = st.columns([1, 3])
            with c1:
                skip = st.checkbox("Prefer not to answer", key=f"{key}__skip")
            if skip:
                A[key] = None
                st.caption("↳ Skipped (blank for the model)")
                return
            v = st.slider(label, 0, 10, 0, key=key)
            A[key] = int(v)

        zero_ten_with_skip("Anxiety score (0–10)", "anxiety_score")
        zero_ten_with_skip("Conscious movement score (0–10)", "conscious_movement_score")

        A["cdte"] = yesno_with_skip(
            "Clock-drawing done today?",
            "cdte",
            desc="Draw an analog clock showing 10 past 11; choose ‘Yes’ if performed now."
        )
        A["cogdt"] = yesno_with_skip(
            "Orientation item done today (date & location)?",
            "cogdt",
            desc="Answer a quick date and location question today, if done."
        )

        # L2 motor/usability items
        st.subheader("Fine motor / function")
        A["TRBBUTTN_OL"] = likert_with_skip("Buttons/clasps difficult — severity", "TRBBUTTN_OL")
        A["TRBUPCHR_OL"] = likert_with_skip("Trouble using phone/computer — severity", "TRBUPCHR_OL")
        A["WRTSMLR_OL"] = likert_with_skip("Handwriting smaller — severity", "WRTSMLR_OL")
        A["DFCLTYTYPE_OL"] = likert_with_skip("Difficulty typing — severity", "DFCLTYTYPE_OL")

        c1, c2 = st.columns([1,1])
        with c1:
            back = st.form_submit_button("Back", use_container_width=True)
        with c2:
            nextp = st.form_submit_button("Next", use_container_width=True)
    if back:
        goto(page - 1)
    if nextp:
        goto(page + 1)

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

    # Predict per layer with diagnostics (now respects zero-input exclusion)
    p1, f1, d1 = safe_predict("Layer 1", MODEL_L1, A)
    p2, f2, d2 = safe_predict("Layer 2", MODEL_L2, A)
    p3, f3, d3 = safe_predict("Layer 3", MODEL_L3, A)

    with st.expander("Diagnostics"):
        st.write(d1)
        st.write(d2)
        st.write(d3)
        if f1: st.caption(f"Layer 1 expected features: {f1}")
        if f2: st.caption(f"Layer 2 expected features: {f2}")
        if f3: st.caption(f"Layer 3 expected features: {f3}")

    # ---- NEW: renormalize over present layers only
    probs = {"layer1": p1, "layer2": p2, "layer3": p3}
    p_final, renorm_w, missing_layers = blend_probs(W, probs)
    risk_cat, advice = categorize(p_final if p_final is not None else float("nan"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Layer 1 prob", "—" if p1 is None else f"{p1*100:.1f}%")
        if p1 is None:
            st.caption("Insufficient data")
    with c2:
        st.metric("Layer 2 prob", "—" if p2 is None else f"{p2*100:.1f}%")
        if p2 is None:
            st.caption("Insufficient data")
    with c3:
        st.metric("Layer 3 prob", "—" if p3 is None else f"{p3*100:.1f}%")
        if p3 is None:
            st.caption("Insufficient data")

    st.subheader("Final PD risk")
    if p_final is not None and np.isfinite(p_final):
        st.success(f"**{p_final*100:.1f}%**  →  **{risk_cat.upper()}**")
    else:
        st.error("Not enough information to compute a final risk.")

    st.write(advice)

    used_layers = [k for k,v in probs.items() if v is not None and np.isfinite(v)]
    if used_layers:
        renorm_str = ", ".join([f"{k}:{renorm_w[k]:.3f}" for k in used_layers])
    else:
        renorm_str = "—"
    st.caption(
        f"Used layers: {', '.join(used_layers) if used_layers else 'none'} "
        f"(renormalized weights over present layers → {renorm_str}; base W: L1={W['layer1']}, L2={W['layer2']}, L3={W['layer3']})"
    )

    st.divider()
    with st.expander("Show raw feature row (debug)"):
        st.dataframe(df_out)

    # ---- FORM WRAPPER ----
    with st.form("form7"):
        c1, c2 = st.columns([1,1])
        with c1:
            back = st.form_submit_button("Back", use_container_width=True)
        with c2:
            finish = st.form_submit_button("Finish", use_container_width=True, disabled=not st.session_state.agree)
    if back:
        goto(page - 1)
