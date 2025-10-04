# app.py — NeuroScreen multi-layer PD risk tool
# Run locally:  streamlit run app.py

import os, io, json, math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, timezone
from PIL import Image

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="NeuroScreen",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------- GLOBAL STYLE --------------------
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
/* bigger bold question label */
.q-label{font-size:20px;font-weight:700;margin:10px 0 6px 0}
</style>
"""
st.markdown(HERO_CSS, unsafe_allow_html=True)

# --------------------- Models -------------------------
MODEL_L1 = "model_layer1.pkl"
MODEL_L2 = "model_layer2_ppmi.pkl"
MODEL_L3 = "model_layer3.pkl"

# Ensemble weights & thresholds
W = {"layer1": 0.2, "layer2": 0.4, "layer3": 0.4}
LOW_THR, HIGH_THR = 0.33, 0.66

# --------------------- Local data folder (fallback) ---
DATA_DIR = Path(os.environ.get("NEURO_DATA_DIR", str(Path.home() / ".neuroscreen_data")))
USERS_CSV = DATA_DIR / "users.csv"
RESULTS_CSV = DATA_DIR / "results.csv"

USER_COLS = ["user_id","name","address","created_at_utc","created_at_local"]
RESULT_COLS = [
    "result_id","user_id","name","address",
    "final_prob","risk_category","layer1_prob","layer2_prob","layer3_prob",
    "used_layers","created_at_utc","created_at_local"
]

def _ensure_data_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_CSV.exists():
        pd.DataFrame(columns=USER_COLS).to_csv(USERS_CSV, index=False)
    if not RESULTS_CSV.exists():
        pd.DataFrame(columns=RESULT_COLS).to_csv(RESULTS_CSV, index=False)

def _normalize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c not in ("final_prob","layer1_prob","layer2_prob","layer3_prob") else np.nan
    return df[cols]

# --------------------- Google Sheets helpers ----------
USE_SHEETS = False
GS_USERS_TAB = "users"
GS_RESULTS_TAB = "results"

try:
    import gspread
    from google.oauth2.service_account import Credentials
    # secrets: SERVICE_ACCOUNT_JSON (full JSON), GSHEET_URL, GSHEET_USERS_TAB, GSHEET_RESULTS_TAB
    if "SERVICE_ACCOUNT_JSON" in st.secrets and "GSHEET_URL" in st.secrets:
        svc_info = json.loads(st.secrets["SERVICE_ACCOUNT_JSON"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(svc_info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(st.secrets["GSHEET_URL"])
        GS_USERS_TAB = st.secrets.get("GSHEET_USERS_TAB", GS_USERS_TAB)
        GS_RESULTS_TAB = st.secrets.get("GSHEET_RESULTS_TAB", GS_RESULTS_TAB)
        USE_SHEETS = True

        def _ensure_worksheet(title: str, header: List[str]):
            try:
                ws = sh.worksheet(title)
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(title=title, rows="1000", cols=str(max(5, len(header))))
                ws.append_row(header)
            # if empty, ensure header row exists
            if ws.row_count == 1 and not ws.get_values("A1:A1"):
                ws.append_row(header)
            return ws

        def _append_dict(title: str, header: List[str], row_dict: Dict[str, Any]):
            ws = _ensure_worksheet(title, header)
            row = [row_dict.get(h, "") for h in header]
            ws.append_row(row, value_input_option="RAW")

        def _read_all(title: str, header: List[str]) -> pd.DataFrame:
            ws = _ensure_worksheet(title, header)
            values = ws.get_all_values()
            if not values:
                return pd.DataFrame(columns=header)
            # ensure header first row
            if values[0] != header:
                # try to align by header names if the sheet already had data
                data_rows = values[1:] if len(values) > 1 else []
                df = pd.DataFrame(data_rows, columns=values[0])
                for c in header:
                    if c not in df.columns:
                        df[c] = ""
                return df[header]
            data_rows = values[1:] if len(values) > 1 else []
            return pd.DataFrame(data_rows, columns=header)
    else:
        USE_SHEETS = False
except Exception:
    USE_SHEETS = False

# --------------------- Data IO (Sheet or CSV) ---------
def load_users() -> pd.DataFrame:
    if USE_SHEETS:
        try:
            df = _read_all(GS_USERS_TAB, USER_COLS)
            return _normalize(df, USER_COLS)
        except Exception:
            pass
    _ensure_data_files()
    try:
        return _normalize(pd.read_csv(USERS_CSV), USER_COLS)
    except Exception:
        return pd.DataFrame(columns=USER_COLS)

def load_results() -> pd.DataFrame:
    if USE_SHEETS:
        try:
            df = _read_all(GS_RESULTS_TAB, RESULT_COLS)
            # convert numeric columns if possible
            for c in ["final_prob","layer1_prob","layer2_prob","layer3_prob"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return _normalize(df, RESULT_COLS)
        except Exception:
            pass
    _ensure_data_files()
    try:
        return _normalize(pd.read_csv(RESULTS_CSV), RESULT_COLS)
    except Exception:
        return pd.DataFrame(columns=RESULT_COLS)

def save_user(name: str, address: str) -> str:
    uid = datetime.now().strftime("%y%m%d%H%M%S%f")[-10:]
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    now_loc = datetime.now().astimezone().isoformat(timespec="seconds")
    row = {"user_id": uid, "name": name.strip(), "address": address.strip(),
           "created_at_utc": now_utc, "created_at_local": now_loc}

    # Sheets first (if enabled)
    if USE_SHEETS:
        try:
            _append_dict(GS_USERS_TAB, USER_COLS, row)
        except Exception as e:
            st.warning(f"Sheets write failed; saved locally instead. ({e})")
            _ensure_data_files()
            df = load_users()
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(USERS_CSV, index=False)
    else:
        _ensure_data_files()
        df = load_users()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(USERS_CSV, index=False)

    return uid

def save_result(user_id: str, name: str, address: str,
                final_prob: Optional[float], risk_category: str,
                p1: Optional[float], p2: Optional[float], p3: Optional[float],
                used_layers: List[str]) -> str:
    rid = datetime.now().strftime("%y%m%d%H%M%S%f")
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    now_loc = datetime.now().astimezone().isoformat(timespec="seconds")
    row = {
        "result_id": rid,
        "user_id": user_id or "",
        "name": (name or "").strip(),
        "address": (address or "").strip(),
        "final_prob": ("" if final_prob is None else float(final_prob)),
        "risk_category": risk_category,
        "layer1_prob": ("" if p1 is None else float(p1)),
        "layer2_prob": ("" if p2 is None else float(p2)),
        "layer3_prob": ("" if p3 is None else float(p3)),
        "used_layers": ",".join(used_layers) if used_layers else "",
        "created_at_utc": now_utc,
        "created_at_local": now_loc,
    }

    if USE_SHEETS:
        try:
            _append_dict(GS_RESULTS_TAB, RESULT_COLS, row)
        except Exception as e:
            st.warning(f"Sheets write failed; saved locally instead. ({e})")
            _ensure_data_files()
            df = load_results()
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(RESULTS_CSV, index=False)
    else:
        _ensure_data_files()
        df = load_results()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(RESULTS_CSV, index=False)

    return rid

# --------------------- UI Helpers ---------------------
def _q_label(label: str, desc: Optional[str]):
    """Render big bold label with helper in parentheses."""
    help_part = f" ({desc})" if desc else ""
    st.markdown(f"<div class='q-label'><strong>{label}</strong>{help_part}</div>", unsafe_allow_html=True)

def likert_with_skip(label, key, desc=None, help_text=None):
    _q_label(label, desc)
    col_slider, col_skip = st.columns([4, 1])
    with col_slider:
        val = st.slider(" ", 0, 4, 0, key=key, help=help_text, label_visibility="collapsed")
    with col_skip:
        skip = st.checkbox("Prefer not to answer", key=f"{key}__skip")
    return None if skip else int(val)

def number_with_skip(label, key, placeholder="", desc=None):
    _q_label(label, desc)
    col_inp, col_skip = st.columns([4, 1])
    with col_inp:
        txt = st.text_input(" ", key=key, placeholder=placeholder, label_visibility="collapsed").strip()
    with col_skip:
        skip = st.checkbox("Prefer not to answer", key=f"{key}__skip")
    if skip or txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        st.warning("Please enter a number (or check 'Prefer not to answer').")
        return None


def yesno_with_skip(label: str, key: str, desc: str=None, help_text: Optional[str]=None) -> Optional[int]:
    _q_label(label, desc)
    choice = st.radio(" ", ["Yes","No","Prefer not to answer"], key=key, horizontal=True, help=help_text, label_visibility="collapsed")
    if choice == "Yes": return 1
    if choice == "No":  return 0
    return None

def sex_radio(key: str, desc: str=None) -> Optional[int]:
    _q_label("Sex", desc)
    choice = st.radio(" ", ["Male","Female","Prefer not to answer"], key=key, horizontal=True, label_visibility="collapsed")
    if choice == "Male": return 1
    if choice == "Female": return 0
    return None


def section_scale_hint():
    st.caption("Scale: 0=None, 1=Slight, 2=Mild, 3=Moderate, 4=Severe. Use 'Prefer not to answer' if unsure.")

# --------------------- Model helpers ------------------
def unwrap_model(obj):
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"], obj.get("features")
    return obj, None

def expected_features(model, feats_hint: Optional[List[str]]) -> Optional[List[str]]:
    if feats_hint:
        return [str(c) for c in feats_hint]
    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]
    return None

def make_single_row_df(answers: Dict[str, Any], feature_list: List[str]) -> pd.DataFrame:
    row = {f: (np.nan if answers.get(f, None) is None else answers.get(f)) for f in feature_list}
    return pd.DataFrame([row], columns=feature_list)

def predict_layer(model_path: str, answers: Dict[str, Any]) -> Tuple[Optional[float], Optional[List[str]]]:
    p = Path(model_path)
    if not p.exists(): return None, None
    bundle = joblib.load(p)
    model, feats_hint = unwrap_model(bundle)
    feats = expected_features(model, feats_hint) or list(answers.keys())
    nn = sum(answers.get(f, None) is not None for f in feats)
    if nn == 0: return None, feats
    X = make_single_row_df(answers, feats).apply(pd.to_numeric, errors="coerce")
    try:
        prob = float(model.predict_proba(X)[:, 1][0])
        return max(0.0, min(1.0, prob)), feats
    except Exception:
        return None, feats

def blend_probs(weights: Dict[str,float], probs: Dict[str,Optional[float]]
               ) -> Tuple[Optional[float], Dict[str,float], List[str]]:
    present = {k: p for k, p in probs.items() if p is not None and np.isfinite(p)}
    missing = [k for k,p in probs.items() if p is None or not np.isfinite(p)]
    if not present: return None, {}, missing
    total_w = sum(weights[k] for k in present.keys())
    renorm = {k: (weights[k]/total_w if total_w>0 else 1/len(present)) for k in present.keys()}
    p_final = sum(renorm[k]*present[k] for k in present.keys())
    return float(p_final), renorm, missing

def categorize(p_final: float) -> Tuple[str,str]:
    if p_final is None or not np.isfinite(p_final):
        return "unknown","Not enough information to compute risk. Please answer a few more items."
    if p_final < LOW_THR:   return "low","Low risk: healthy habits, regular exercise, and routine check-ups are advisable."
    if p_final < HIGH_THR:  return "medium","Moderate risk: consider a neurological evaluation if symptoms persist or worsen."
    return "high","High risk: please seek a clinical evaluation from a neurologist specializing in movement disorders."

def model_exists(path):
    p = Path(path); return p.exists(), str(p.resolve())

def nonnull_count_for(features, answers): return sum(answers.get(f, None) is not None for f in (features or []))

def safe_predict(name, path, answers):
    ok, full = model_exists(path)
    if not ok: return None, None, f"{name}: model file missing at {full}"
    try:
        prob, feats = predict_layer(path, answers)
        if prob is None:
            nn = nonnull_count_for(feats, answers)
            if nn == 0: return None, feats, f"{name}: Insufficient data (0 usable inputs)"
            return None, feats, f"{name}: predict_proba failed or returned None (nonnull features fed: {nn})"
        return prob, feats, f"{name}: OK (nonnull features fed: {nonnull_count_for(feats, answers)})"
    except Exception as e:
        return None, None, f"{name}: exception -> {e}"

# --------------------- Session state -------------------
if "page" not in st.session_state: st.session_state.page = 0
if "agree" not in st.session_state: st.session_state.agree = False
if "answers" not in st.session_state: st.session_state.answers = {}
if "user_id" not in st.session_state: st.session_state.user_id = ""
if "user_name" not in st.session_state: st.session_state.user_name = ""
if "user_addr" not in st.session_state: st.session_state.user_addr = ""

# Show/hide Sign-Ups page
SHOW_SIGNUPS_PAGE = False

# --------------------- Pages --------------------------
PAGES = [
    "Sign Up",
    "Consent",
    "Demographics",
    "Smell & Bowel",
    "Urinary & Saliva",
    "Tremor & Balance",
    "Fatigue & Dizziness",
    "Memory & Anxiety",
    "Review & Results",
] + (["Sign-Ups"] if SHOW_SIGNUPS_PAGE else [])

def goto_index(i: int):
    i = int(np.clip(i, 0, len(PAGES) - 1))
    st.session_state.page = i
    st.session_state["nav_force_sync"] = True

def go_next():
    goto_index(st.session_state.page + 1)
    st.rerun()

def go_back():
    goto_index(st.session_state.page - 1)
    st.rerun()

# --------------------- Sidebar ------------------------
if st.session_state.get("nav_force_sync", False):
    st.session_state["nav_radio"] = PAGES[st.session_state.page]
    st.session_state["nav_force_sync"] = False

def _sync_page_from_radio(): st.session_state.page = PAGES.index(st.session_state["nav_radio"])

st.sidebar.title("NeuroScreen")
st.sidebar.subheader("Navigate")

# Sign-up count (Sheets if available; else CSV)
try:
    if USE_SHEETS:
        df_users = load_users()
        _cnt = len(df_users)
    else:
        _ensure_data_files()
        _cnt = len(load_users())
except Exception:
    _cnt = 0
st.sidebar.metric("Sign-ups", _cnt)

st.sidebar.radio("Pages", PAGES, key="nav_radio", index=st.session_state.page, on_change=_sync_page_from_radio)
st.sidebar.caption(f"Storage: {'Google Sheets' if USE_SHEETS else str(DATA_DIR)}")

# --------------------- Header -------------------------
LOCAL_HEADER = Path("PIC.png")
if LOCAL_HEADER.exists():
    img = Image.open(LOCAL_HEADER); st.image(img, use_column_width=True)
else:
    st.warning("Local header image not found: PIC.png")

if st.session_state.user_name:
    st.info(f"Signed in as **{st.session_state.user_name}**")

st.markdown("### HIPAA Notice & Consent")
st.write(
    "HIPAA Privacy Notice: This tool collects health-related answers for the limited purpose of risk triage and research. "
    "Data are stored only for the current session unless you choose to export. Do not enter personally identifiable information. "
    "By checking 'I Agree', you consent to use of your answers for the risk estimate shown in this app and to aggregate anonymized analytics."
)
st.session_state.agree = st.checkbox("I Agree", value=st.session_state.agree)

st.markdown(f"**Progress:** {' > '.join(PAGES)}")

A = st.session_state.answers
page = st.session_state.page
page_name = PAGES[page]

# --------------------- Pages --------------------------
# Sign Up
if page_name == "Sign Up":
    st.header("Create an account")
    st.caption("Become part of the family")

    with st.form("signup_form", clear_on_submit=False):
        name = st.text_input("Name", key="signup_name", placeholder="e.g., Alex Johnson")
        address = st.text_input("Address (City/State or any label)", key="signup_addr", placeholder="e.g., St. Louis, MO")
        st.caption("Password field omitted — sign-ups are counted without authentication.")
        c1, c2, c3 = st.columns([1,1,1])
        with c1: back = st.form_submit_button("Back", use_container_width=True)
        with c2: submit = st.form_submit_button("Sign Up", use_container_width=True)
        with c3: nextp = st.form_submit_button("Next → Consent", use_container_width=True)

    if back: go_back()

    if submit:
        if not name.strip() or not address.strip():
            st.error("Please enter both your name and address.")
        else:
            uid = save_user(name=name, address=address)
            st.session_state.user_id = uid
            st.session_state.user_name = name.strip()
            st.session_state.user_addr = address.strip()
            st.success(f"Thanks for signing up, **{name}**! Your ID is `{uid}`.")
            st.rerun()

    # Next auto-signs up if needed (so everyone is counted)
    if nextp:
        if st.session_state.user_id:
            go_next()
        elif name.strip() and address.strip():
            uid = save_user(name=name, address=address)
            st.session_state.user_id = uid
            st.session_state.user_name = name.strip()
            st.session_state.user_addr = address.strip()
            st.success(f"Signed up and continuing. ID `{uid}`.")
            go_next()
        else:
            st.error("Please enter Name and Address (or click Sign Up) before continuing.")

# Consent
elif page_name == "Consent":
    st.header("Consent")
    st.info("Please confirm consent to proceed. You can still browse, but 'Finish & Save Result' requires consent.")
    with st.form("form0"):
        c1, c2 = st.columns([1,1])
        with c1: back = st.form_submit_button("Back", use_container_width=True)
        with c2: nextp = st.form_submit_button("Next", disabled=(not st.session_state.agree), use_container_width=True)
    if back: go_back()
    if nextp: go_next()

# Demographics
elif page_name == "Demographics":
    st.header("Demographics")
    with st.form("form1"):
        age = number_with_skip("Age (years)", key="age_num", desc="Enter your age in whole years.")
        sex_val = sex_radio("sex_radio", desc="Select your sex.")
        A["age"] = age; A["AGE_AT_VISIT"] = age; A["screening_age"] = age
        A["Sex"] = sex_val; A["SEX"] = sex_val; A["sex"] = sex_val
        c1, c2 = st.columns([1,1])
        with c1: back = st.form_submit_button("Back", use_container_width=True)
        with c2: nextp = st.form_submit_button("Next", use_container_width=True)
    if back: go_back()
    if nextp: go_next()

# Smell & Bowel
elif page_name == "Smell & Bowel":
    st.header("Smell & Bowel")
    section_scale_hint()
    with st.form("form2"):
        A["ImpactMoveHyposmia"] = likert_with_skip("Reduced sense of smell — severity","ImpactMoveHyposmia",
                                                   desc="Sniff coffee grounds or soap and rate how strong the smell is.")
        A["ImpactMoveConstipation"] = likert_with_skip("Constipation affecting daily life — severity","ImpactMoveConstipation",
                                                       desc="Past 2 weeks: hard stools, straining, or fewer than 3 bowel movements per week.")
        A["UPSIT_PRESENT"] = yesno_with_skip("Reduced smell present (Yes/No)?","UPSIT_PRESENT",
                                             desc="If coffee/soap/lotion smells faint or absent, choose ‘Yes’.")
        c1, c2 = st.columns([1,1])
        with c1: back = st.form_submit_button("Back", use_container_width=True)
        with c2: nextp = st.form_submit_button("Next", use_container_width=True)
    if back: go_back()
    if nextp: go_next()

# Urinary & Saliva
elif page_name == "Urinary & Saliva":
    st.header("Urinary & Saliva")
    section_scale_hint()
    with st.form("form3"):
        A["ImpactMoveUrinary"] = likert_with_skip("Urinary urgency/frequency — severity","ImpactMoveUrinary",
                                                  desc="Do you rush to the bathroom or go more often than usual?")
        A["MoveSaliva"] = likert_with_skip("Excess saliva/drooling — severity","MoveSaliva",
                                           desc="Do you need to swallow often or notice drool on your pillow?")
        A["URIN_PRESENT"] = yesno_with_skip("Urinary problems present (Yes/No)?","URIN_PRESENT",
                                            desc="Urgency, frequency, or nighttime bathroom trips more than twice.")
        A["NP1URIN_OL"] = likert_with_skip("Urinary difficulties (overall) — severity","NP1URIN_OL",
                                           desc="Overall rating of urinary difficulties in the past 2 weeks.")
        A["PQUEST_SOURCE_OL"] = yesno_with_skip("Did you complete these questions yourself? (Yes/No)","PQUEST_SOURCE_OL",
                                                desc="Choose ‘Yes’ if you personally answered these questions today.")
        c1, c2 = st.columns([1,1])
        with c1: back = st.form_submit_button("Back", use_container_width=True)
        with c2: nextp = st.form_submit_button("Next", use_container_width=True)
    if back: go_back()
    if nextp: go_next()

# Tremor & Balance
elif page_name == "Tremor & Balance":
    st.header("Tremor & Balance")
    section_scale_hint()
    with st.form("form4"):
        A["ImpactMoveTremor"] = likert_with_skip("Tremor affecting tasks — severity","ImpactMoveTremor",
                                                 desc="Hold your hands out for 10 seconds or try lifting a cup; rate the impact.")
        A["ARMLGSHK_OL"] = likert_with_skip("Arm/leg shaking (tremor) — severity","ARMLGSHK_OL",
                                            desc="Shaking in the arms or legs at rest or during action; rate typical severity.")
        A["SHUFFLE_OL"] = likert_with_skip("Shuffling steps — severity","SHUFFLE_OL",
                                           desc="Short, shuffling steps when walking; rate how often/severe this feels.")
        A["MVSLOW_OL"] = likert_with_skip("Slowness of movement — severity","MVSLOW_OL",
                                          desc="General slowness (bradykinesia): smaller/fewer movements or slower actions.")
        A["POORBAL_OL"] = likert_with_skip("Poor balance / near-falls — severity","POORBAL_OL",
                                           desc="Unsteady balance, near-falls, or needing support to steady yourself.")
        A["FTSTUCK_OL"] = likert_with_skip("Feet feel stuck / freezing — severity","FTSTUCK_OL",
                                           desc="Feet ‘stick’ when starting, turning, or in narrow spaces; rate frequency/severity.")
        st.subheader("Walking ability (short test)")
        st.caption("If safe, try 10 meters. If you use a cane/walker, select that option. Skip if unsafe.")
        walk_choice = st.radio("Can you walk across a room unaided?", ["Yes","With aid","No","Prefer not to answer"],
                               horizontal=True, key="screening_walk_radio")
        if      walk_choice == "Yes":      A["screening_walk"] = 1.0
        elif    walk_choice == "With aid": A["screening_walk"] = 0.5
        elif    walk_choice == "No":       A["screening_walk"] = 0.0
        else:                               A["screening_walk"] = None
        A["dt_tug_time"] = number_with_skip("Timed Up & Go (seconds)", key="dt_tug_time", placeholder="e.g., 9.6",
                                            desc="From seated: stand, walk 3 m, turn, return, and sit; enter the total seconds.")
        c1, c2 = st.columns([1,1])
        with c1: back = st.form_submit_button("Back", use_container_width=True)
        with c2: nextp = st.form_submit_button("Next", use_container_width=True)
    if back: go_back()
    if nextp: go_next()

# Fatigue & Dizziness
elif page_name == "Fatigue & Dizziness":
    st.header("Fatigue & Dizziness")
    section_scale_hint()
    with st.form("form5"):
        A["ImpactThinkBodyFatigue"] = likert_with_skip("Fatigue slowing thinking/activity — severity","ImpactThinkBodyFatigue",
                                                       desc="Is daytime energy low; do tasks feel slower than usual?")
        A["ImpactThinkDizzy"] = likert_with_skip("Dizziness / light-headedness — severity","ImpactThinkDizzy",
                                                 desc="Stand up and count to 10—did you feel woozy or light-headed?")
        A["VOICSFTR_OL"] = likert_with_skip("Softer / quieter voice — severity","VOICSFTR_OL")
        A["LSSXPRSS_OL"] = likert_with_skip("Reduced facial expression — severity","LSSXPRSS_OL")
        A["RBD_PRESENT"] = yesno_with_skip("Acting out dreams during sleep (kicking/shouting)?","RBD_PRESENT",
                                           desc="Ask a bed partner if unsure; choose ‘Yes’ if dream enactment is present.")
        if A.get("RBD_PRESENT", None) == 1:
            src = st.radio("Who noticed/diagnosed this?", ["Self (1)","Bed partner (2)","Physician (3)","Prefer not to answer"],
                           horizontal=True, key="RBD_SOURCE_OL_radio")
            A["RBD_SOURCE_OL"] = {"Self (1)":1,"Bed partner (2)":2,"Physician (3)":3}.get(src, None)
        else:
            A["RBD_SOURCE_OL"] = None
        c1, c2 = st.columns([1,1])
        with c1: back = st.form_submit_button("Back", use_container_width=True)
        with c2: nextp = st.form_submit_button("Next", use_container_width=True)
    if back: go_back()
    if nextp: go_next()

# Memory & Anxiety
elif page_name == "Memory & Anxiety":
    st.header("Memory & Anxiety")
    section_scale_hint()

    with st.form("form6"):
        # --- Core items ---
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

        # --- Quick cognitive checks ---
        st.subheader("Quick cognitive checks")
        A["DIFFRECALL"] = likert_with_skip(
            "Word recall difficulty — severity",
            "DIFFRECALL",
            desc="Say 3 words; after 1 minute, try to recall them."
        )
        A["MCAVIGIL"] = likert_with_skip(
            "Sustained attention issues — severity",
            "MCAVIGIL",
            desc="Count backward by 7s for 30 seconds; rate difficulty."
        )
        A["MEM_PRESENT"] = yesno_with_skip(
            "Memory problems present (Yes/No)?",
            "MEM_PRESENT",
            desc="Misplacing items, repeating questions, or forgetting recent events."
        )

        # --- Short protocol items ---
        st.subheader("Short protocol items")
        def zero_ten_with_skip(label, key):
            _q_label(label, None)  # big bold label (no helper text here)
            col_s, col_skip = st.columns([4, 1])
            with col_s:
                v = st.slider(" ", 0, 10, 0, key=key, label_visibility="collapsed")
            with col_skip:
                skip = st.checkbox("Prefer not to answer", key=f"{key}__skip")
            A[key] = (None if skip else int(v))

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

        # --- Fine motor / function ---
        st.subheader("Fine motor / function")
        A["TRBBUTTN_OL"] = likert_with_skip("Buttons/clasps difficult — severity", "TRBBUTTN_OL")
        A["TRBUPCHR_OL"] = likert_with_skip("Trouble using phone/computer — severity", "TRBUPCHR_OL")
        A["WRTSMLR_OL"] = likert_with_skip("Handwriting smaller — severity", "WRTSMLR_OL")
        A["DFCLTYTYPE_OL"] = likert_with_skip("Difficulty typing — severity", "DFCLTYTYPE_OL")

        # ✅ ALWAYS include submit buttons inside the form
        c1, c2 = st.columns([1, 1])
        with c1:
            back = st.form_submit_button("Back", use_container_width=True)
        with c2:
            nextp = st.form_submit_button("Next", use_container_width=True)

    if back:
        go_back()
    if nextp:
        go_next()


# Review & Results
elif page_name == "Review & Results":
    st.header("Review & Results")
    if "id" not in A or not A.get("id"): A["id"] = "A001"

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
    for k in cols_order: A.setdefault(k, None)
    df_out = pd.DataFrame([[A.get(c, None) for c in cols_order]], columns=cols_order)
    df_out.to_csv("app_inputs.csv", index=False)
    st.caption("Saved current answers → app_inputs.csv")

    p1, f1, d1 = safe_predict("Layer 1", MODEL_L1, A)
    p2, f2, d2 = safe_predict("Layer 2", MODEL_L2, A)
    p3, f3, d3 = safe_predict("Layer 3", MODEL_L3, A)

    with st.expander("Diagnostics"):
        st.write(d1); st.write(d2); st.write(d3)
        if f1: st.caption(f"Layer 1 expected features: {f1}")
        if f2: st.caption(f"Layer 2 expected features: {f2}")
        if f3: st.caption(f"Layer 3 expected features: {f3}")

    probs = {"layer1": p1, "layer2": p2, "layer3": p3}
    p_final, renorm_w, missing_layers = blend_probs(W, probs)
    risk_cat, advice = categorize(p_final if p_final is not None else float("nan"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Layer 1 prob", "—" if p1 is None else f"{p1*100:.1f}%")
        if p1 is None: st.caption("Insufficient data")
    with c2:
        st.metric("Layer 2 prob", "—" if p2 is None else f"{p2*100:.1f}%")
        if p2 is None: st.caption("Insufficient data")
    with c3:
        st.metric("Layer 3 prob", "—" if p3 is None else f"{p3*100:.1f}%")
        if p3 is None: st.caption("Insufficient data")

    st.subheader("Final PD risk")
    if p_final is not None and np.isfinite(p_final):
        st.success(f"**{p_final*100:.1f}%**  →  **{risk_cat.upper()}**")
    else:
        st.error("Not enough information to compute a final risk.")
    st.write(advice)

    used_layers = [k for k,v in probs.items() if v is not None and np.isfinite(v)]
    renorm_str = ", ".join([f"{k}:{renorm_w[k]:.3f}" for k in used_layers]) if used_layers else "—"
    st.caption(f"Used layers: {', '.join(used_layers) if used_layers else 'none'} (renormalized weights → {renorm_str})")

    st.divider()
    with st.expander("Show raw feature row (debug)"):
        st.dataframe(df_out)

    with st.form("form7"):
        c1, c2 = st.columns([1,1])
        with c1: back = st.form_submit_button("Back", use_container_width=True)
        with c2: finish = st.form_submit_button("Finish & Save Result", use_container_width=True, disabled=not st.session_state.agree)
    if back: go_back()
    if finish:
        rid = save_result(
            user_id=st.session_state.user_id,
            name=st.session_state.user_name,
            address=st.session_state.user_addr,
            final_prob=(None if p_final is None or not np.isfinite(p_final) else float(p_final)),
            risk_category=risk_cat,
            p1=p1, p2=p2, p3=p3,
            used_layers=used_layers
        )
        st.success(f"Result saved with id `{rid}` → {'Google Sheets' if USE_SHEETS else RESULTS_CSV}")

# Sign-Ups (public list)
elif page_name == "Sign-Ups":
    st.header("All Sign-Ups")
    df = load_users()
    if df.empty:
        st.info("No sign-ups yet.")
    else:
        st.metric("Total sign-ups", len(df))
        df_disp = df.copy()
        try:
            df_disp["__sort"] = pd.to_datetime(df_disp["created_at_utc"], errors="coerce")
            df_disp = df_disp.sort_values("__sort", ascending=False).drop(columns="__sort")
        except Exception:
            pass
        st.dataframe(df_disp[["name","address","created_at_local","created_at_utc","user_id"]],
                     use_container_width=True, hide_index=True)
        if not USE_SHEETS:
            try:
                st.download_button("Download users.csv", data=USERS_CSV.read_bytes(),
                                   file_name="users.csv", mime="text/csv")
            except Exception:
                st.warning("Could not read users.csv for download.")
