# train_layer2_ppmi.py
# PPMI Layer-2 trainer with fixed feature set + calibration + flexible label

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score


# --- Fixed configuration ---
IDCOL = "PATNO"
LABEL_CANDIDATES = ["APPRDX", "CurrPDDiag"]  # try APPRDX first, then CurrPDDiag
FEATURES = [
    "AGE_AT_VISIT","SEX","DIFFRECALL","MCAVIGIL","NP1URIN_OL","RBD_SOURCE_OL",
    "UPSIT_PRESENT","RBD_PRESENT","URIN_PRESENT","MEM_PRESENT","PQUEST_SOURCE_OL",
    "TRBUPCHR_OL","WRTSMLR_OL","DFCLTYTYPE_OL","VOICSFTR_OL","FTSTUCK_OL",
    "LSSXPRSS_OL","ARMLGSHK_OL","TRBBUTTN_OL","SHUFFLE_OL","MVSLOW_OL","POORBAL_OL"
]


def parse_args():
    ap = argparse.ArgumentParser(description="Train PPMI Layer-2 (fixed feature set)")
    ap.add_argument("--csv", required=True, help="Input merged CSV with PATNO, label, features")
    ap.add_argument("--outdir", default="layer2_ppmi_out", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--valid_frac", type=float, default=0.10)
    ap.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="sigmoid",
                    help="Probability calibration method (default sigmoid)")
    return ap.parse_args()


def map_label_to_binary(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.upper().str.strip()
    # treat anything starting with "PD" or numeric "1" as PD=1; "HC"/"0" as HC=0; else NaN
    return t.map(lambda x: 1 if (x.startswith("PD") or x == "1")
                 else (0 if (x == "HC" or x == "0") else np.nan))


def build_models(seed: int):
    return {
        "logreg": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed))
        ]),
        "rf": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=800, min_samples_leaf=3, n_jobs=-1,
                random_state=seed, class_weight="balanced_subsample"))
        ]),
        "hgb": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=seed))
        ]),
    }


def predict_proba01(clf, X):
    # clip to avoid exact 0/1 (helps ROC/Youden)
    p = clf.predict_proba(X)[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def class_count(y):
    u, c = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # --- Load
    df = pd.read_csv(args.csv, low_memory=False)

    # --- Pick label column
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise KeyError(f"Missing label column; tried {LABEL_CANDIDATES}")
    y_series = map_label_to_binary(df[label_col])
    df = df.assign(y=y_series).dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    # --- Ensure ID and features exist
    if IDCOL not in df.columns:
        raise KeyError(f"Missing ID column {IDCOL!r}")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required features: {missing}")

    # --- Assemble X, y, groups
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df["y"].values
    groups = df[IDCOL].astype(str).values

    # --- Grouped split: train / valid / test
    gss1 = GroupShuffleSplit(n_splits=1, train_size=args.train_frac, random_state=args.seed)
    tr_idx, hold_idx = next(gss1.split(X, y, groups=groups))
    valid_frac_adj = args.valid_frac / (1.0 - args.train_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=valid_frac_adj, random_state=args.seed + 1)
    va_idx, te_idx = next(gss2.split(X.iloc[hold_idx], y[hold_idx], groups[hold_idx]))

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[hold_idx].iloc[va_idx], y[hold_idx][va_idx]
    X_te, y_te = X.iloc[hold_idx].iloc[te_idx], y[hold_idx][te_idx]

    print(f"[split] train n={len(y_tr)} class_counts={class_count(y_tr)}")
    print(f"[split] valid n={len(y_va)} class_counts={class_count(y_va)}")
    print(f"[split] test  n={len(y_te)} class_counts={class_count(y_te)}")

    # --- Train base models and pick best by valid AUC
    models = build_models(args.seed)
    results, best_auc, best_name, best_pipe, best_p_va = [], -np.inf, None, None, None

    for name, pipe in models.items():
        print(f"[fit] {name}")
        pipe.fit(X_tr, y_tr)
        p_va = predict_proba01(pipe, X_va)
        p_te = predict_proba01(pipe, X_te)
        auc_v, auc_t = roc_auc_score(y_va, p_va), roc_auc_score(y_te, p_te)
        print(f"      valid AUC={auc_v:.4f} test AUC={auc_t:.4f}")
        results.append({"model": name, "auc_valid": auc_v, "auc_test": auc_t})
        if auc_v > best_auc:
            best_auc, best_name, best_pipe, best_p_va = auc_v, name, pipe, p_va

    # --- Optional probability calibration (on validation set only)
    calibrated = False
    if args.calibrate != "none":
        print(f"[cal] Calibrating best='{best_name}' with {args.calibrate} on validation split...")
        cal = CalibratedClassifierCV(best_pipe, method=args.calibrate, cv="prefit")
        cal.fit(X_va, y_va)
        best_pipe = cal
        best_p_va = predict_proba01(best_pipe, X_va)
        calibrated = True

    # --- Threshold via Youden J (maximize TPR - FPR) on validation
    fpr, tpr, thr = roc_curve(y_va, best_p_va)
    thr_idx = int(np.argmax(tpr - fpr))
    threshold = float(thr[thr_idx])
    print(f"[thr] best={best_name}{' (cal)' if calibrated else ''} threshold={threshold:.3f}")

    # --- Final test evaluation (never used in training or calibration)
    p_te = predict_proba01(best_pipe, X_te)
    y_pred = (p_te >= threshold).astype(int)
    auc_test = roc_auc_score(y_te, p_te)
    acc_test = accuracy_score(y_te, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    print(f"[test] best={best_name}{' (cal)' if calibrated else ''} "
          f"AUC={auc_test:.4f} ACC={acc_test:.4f} Sens={sens:.3f} Spec={spec:.3f}")

    # --- Save artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, outdir / "model_layer2_ppmi.pkl")
    pd.DataFrame(results).to_csv(outdir / "metrics_summary.csv", index=False)
    with open(outdir / "threshold.json", "w") as f:
        json.dump({
            "threshold": threshold,
            "best_model": best_name,
            "calibrated": calibrated,
            "calibration_method": None if not calibrated else args.calibrate,
            "label_col": label_col,
            "source_csv": Path(args.csv).name
        }, f, indent=2)
    pd.DataFrame({"feature": FEATURES}).to_csv(outdir / "ppmi_feature_list.csv", index=False)


if __name__ == "__main__":
    main()
