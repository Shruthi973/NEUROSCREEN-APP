# train_layer3_shortened.py
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

def eval_at_threshold(y_true, probs, thr):
    y_pred = (probs >= thr).astype(int)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return auc, acc, sens, spec, (tn, fp, fn, tp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", default="screening_pd_dx")
    ap.add_argument("--id", default="record_id")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    df.columns = df.columns.str.strip()

    # Label
    y = pd.to_numeric(df[args.label], errors="coerce").fillna(0).astype(int).values

    # Hand-picked features
    selected = [c for c in [
        "screening_age", "sex",
        "anxiety_score", "conscious_movement_score", "screening_walk",
        "cdte", "cogdt", "dt_tug_time"
    ] if c in df.columns]
    if not selected:
        raise SystemExit("No usable features found!")

    X = df[selected].copy()

    # 50/20/30 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42, stratify=y_temp)
    print(f"[data] train={len(X_train)} valid={len(X_valid)} test={len(X_test)}")

    # Model: RF
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), selected)])
    rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", rf)])
    pipe.fit(X_train, y_train)

    # --- FIXED THRESHOLD 0.50 ---
    thr = 0.50

    # Final test metrics
    p_test = pipe.predict_proba(X_test)[:, 1]
    auc, acc, sens, spec, cm = eval_at_threshold(y_test, p_test, thr)
    print(f"[fixed threshold=0.50]")
    print(f"[test] AUC={auc:.3f}  ACC={acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")
    print(f"[cm]   TN={cm[0]} FP={cm[1]} FN={cm[2]} TP={cm[3]}")

    # Save outputs
    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(pipe, f"{args.outdir}/model_layer3_shortened.pkl")
    with open(f"{args.outdir}/threshold.json", "w") as f:
        json.dump({"threshold": thr}, f)
    pd.DataFrame({"feature": selected}).to_csv(f"{args.outdir}/layer3_feature_list.csv", index=False)
    pd.DataFrame({"id": df[args.id], "p_layer3": pipe.predict_proba(X)[:,1]}).to_csv(
        f"{args.outdir}/p_layer3.csv", index=False)

    print(f"✅ Saved model, metrics, features, threshold=0.50, probs -> {args.outdir}")

if __name__ == "__main__":
    main()
