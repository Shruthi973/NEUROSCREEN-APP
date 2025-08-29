# train_layer2_ppmi.py
# PPMI Layer-2 trainer with fixed feature set

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score

# --- Fixed configuration ---
IDCOL = "PATNO"
LABEL = "APPRDX"
FEATURES = [
    "AGE_AT_VISIT","SEX","DIFFRECALL","MCAVIGIL","NP1URIN_OL","RBD_SOURCE_OL",
    "UPSIT_PRESENT","RBD_PRESENT","URIN_PRESENT","MEM_PRESENT","PQUEST_SOURCE_OL",
    "TRBUPCHR_OL","WRTSMLR_OL","DFCLTYTYPE_OL","VOICSFTR_OL","FTSTUCK_OL",
    "LSSXPRSS_OL","ARMLGSHK_OL","TRBBUTTN_OL","SHUFFLE_OL","MVSLOW_OL","POORBAL_OL"
]

def parse_args():
    ap = argparse.ArgumentParser(description="Train PPMI Layer-2 (fixed feature set)")
    ap.add_argument("--csv", required=True, help="Input merged CSV with PATNO, APPRDX, features")
    ap.add_argument("--outdir", default="layer2_ppmi_out", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--valid_frac", type=float, default=0.10)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # --- Label: APPRDX (1=PD, 0=HC, drop others)
    y = df[LABEL].map(lambda x: 1 if str(x).upper().startswith("PD") or str(x)=="1" else (0 if str(x).upper()=="HC" or str(x)=="0" else np.nan))
    df["y"] = y
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    # --- Features
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    groups = df[IDCOL].astype(str).values
    y = df["y"].values

    # --- Split (grouped by PATNO)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=args.train_frac, random_state=args.seed)
    tr_idx, hold_idx = next(gss1.split(X, y, groups=groups))
    valid_frac_adj = args.valid_frac / (1.0 - args.train_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=valid_frac_adj, random_state=args.seed+1)
    va_idx, te_idx = next(gss2.split(X.iloc[hold_idx], y[hold_idx], groups[hold_idx]))

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[hold_idx].iloc[va_idx], y[hold_idx][va_idx]
    X_te, y_te = X.iloc[hold_idx].iloc[te_idx], y[hold_idx][te_idx]

    # --- Models
    models = {
        "logreg": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=args.seed))
        ]),
        "rf": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=800, min_samples_leaf=3, n_jobs=-1,
                random_state=args.seed, class_weight="balanced_subsample"))
        ]),
        "hgb": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=args.seed))
        ]),
    }

    # --- Train & pick best
    results, best_auc, best_name, best_pipe, best_p_va = [], -np.inf, None, None, None
    for name, pipe in models.items():
        print(f"[fit] {name}")
        pipe.fit(X_tr, y_tr)
        p_va = pipe.predict_proba(X_va)[:, 1]
        p_te = pipe.predict_proba(X_te)[:, 1]
        auc_v, auc_t = roc_auc_score(y_va, p_va), roc_auc_score(y_te, p_te)
        print(f"      valid AUC={auc_v:.4f} test AUC={auc_t:.4f}")
        results.append({"model": name, "auc_valid": auc_v, "auc_test": auc_t})
        if auc_v > best_auc:
            best_auc, best_name, best_pipe, best_p_va = auc_v, name, pipe, p_va

    # --- Threshold via Youden J
    fpr, tpr, thr = roc_curve(y_va, best_p_va)
    thr_idx = int(np.argmax(tpr - fpr))
    threshold = float(thr[thr_idx])
    print(f"[thr] best={best_name} threshold={threshold:.3f}")

    # --- Final test eval
    p_te = best_pipe.predict_proba(X_te)[:, 1]
    y_pred = (p_te >= threshold).astype(int)
    auc_test = roc_auc_score(y_te, p_te)
    acc_test = accuracy_score(y_te, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    print(f"[test] best={best_name} AUC={auc_test:.4f} ACC={acc_test:.4f} Sens={sens:.3f} Spec={spec:.3f}")

    # --- Save
    joblib.dump(best_pipe, outdir/"model_layer2_ppmi.pkl")
    pd.DataFrame(results).to_csv(outdir/"metrics_summary.csv", index=False)
    with open(outdir/"threshold.json","w") as f:
        json.dump({"threshold": threshold, "best_model": best_name}, f, indent=2)
    pd.DataFrame({"feature": FEATURES}).to_csv(outdir/"ppmi_feature_list.csv", index=False)

if __name__ == "__main__":
    main()
