# train_layer1.py
import warnings, json
import numpy as np, pandas as pd, cloudpickle
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore", category=FutureWarning)

# ======== CONFIG ========
DATA   = "FoxInsight.csv"      # merged clinical CSV you just created
IDCOL  = "fox_insight_id"
LABEL  = "CurrPDDiag"

# Auto-selected columns from your last merge run
AUTO_COLS = {
  "hyposmia":             "ImpactMoveHyposmia",
  "constipation":         "ImpactMoveConstipation",
  "depression_anxiety":   "Anxiety",
  "urinary_dysfunction":  "ImpactMoveUrinary",
  "erectile_dysfunction": "days_elapsed",            # suspicious → will be dropped
  "caffeine_intake":      "ProstateAge",             # suspicious → will be dropped
  "head_injury":          "SexAtBirth",              # suspicious → will be dropped
  "family_history_pd":    "MindTMOther",             # suspicious → will be dropped
  "tremor":               "ImpactMoveTremor",
  "fatigue":              "ImpactThinkBodyFatigue",
  "dizziness":            "ImpactThinkDizzy",
  "balance_problems":     "ImpactMoveImbalance",
  "memory_problems":      "ImpactThinkMemory",
  "hypersalivation":      "MoveSaliva",
  "age":                  "age",
  "gender":               "Sex",
}

# obvious string-based guards to drop bad mappings (name contains any of these)
SUSPICIOUS_BY_FEATURE = {
  "erectile_dysfunction": ["days_elapsed","age","sex","gender","sleep","snore","prostate"],
  "caffeine_intake":      ["prostate","age","sex","gender","bp","blood","injury"],
  "head_injury":          ["sexatbirth","sex","gender","age","prostate"],
  "family_history_pd":    ["mindtmother","other","stress","survey","mood"]
}

# ======== HELPERS ========
def to_num(v):
    s = str(v).strip().lower()
    if s in {"1","yes","y","true","present"}: return 1.0
    if s in {"2","no","n","false","absent"}:  return 0.0
    if s in {"", "nan", "none", "unknown", "don’t know", "dont know", "dk"}: return np.nan
    try:
        return float(s)
    except:
        return np.nan

# ======== MAIN ========
def main():
    df = pd.read_csv(DATA, low_memory=False)
    need = {IDCOL, LABEL}
    if not need.issubset(df.columns):
        raise SystemExit(f"❌ {DATA} must contain {sorted(need)}")

    # start from AUTO_COLS but keep only those present
    feat_map = {k:v for k,v in AUTO_COLS.items() if v in df.columns}

    # drop suspicious picks
    dropped = []
    for feat, col in list(feat_map.items()):
        low = col.lower()
        if feat in SUSPICIOUS_BY_FEATURE and any(bad in low for bad in SUSPICIOUS_BY_FEATURE[feat]):
            dropped.append((feat,col))
            feat_map.pop(feat, None)

    if dropped:
        print("⚠️ Auto-dropped suspicious mappings:")
        for feat,col in dropped:
            print(f"   - {feat} → {col}")

    use_cols = list(feat_map.values())
    if not use_cols:
        raise SystemExit("❌ No usable feature columns left after filtering.")

    # build X
    X = df.loc[:, use_cols].copy()
    for c in X.columns:
        X.loc[:, c] = X[c].map(to_num)

    # map Sex if needed
    if "Sex" in X.columns and not np.issubdtype(X["Sex"].dropna().infer_objects().dtype, np.number):
        X.loc[:, "Sex"] = X["Sex"].astype(str).str[:1].str.lower().map({"m":1, "f":0})

    # label
    y = df[LABEL].astype(float)

    # rows with label + at least one feature
    mask = (~y.isna()) & (X.notna().sum(axis=1) > 0)
    ids  = df.loc[mask, IDCOL].values
    X, y = X[mask], y[mask]

    # simple impute
    X = X.fillna(X.median(numeric_only=True))

    # split for honest report
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # elastic-net logistic (saga) + isotonic calibration
    base = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.35,
        C=1.0
    )
    clf = CalibratedClassifierCV(base, cv=5, method="isotonic")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]

    print("\nStage-1 Clinical Model")
    print("  AUROC:", round(roc_auc_score(yte, p), 3))
    print("  AUPRC:", round(average_precision_score(yte, p), 3))
    print("  Brier:", round(brier_score_loss(yte, p), 3))

    # fit on ALL for production
    clf_full = CalibratedClassifierCV(
        LogisticRegression(
            max_iter=5000, class_weight="balanced",
            penalty="elasticnet", solver="saga", l1_ratio=0.35, C=1.0
        ),
        cv=5, method="isotonic"
    )
    clf_full.fit(X, y)
    p_all = clf_full.predict_proba(X)[:,1]

    # export per-patient probabilities
    pd.DataFrame({IDCOL: ids, "p_layer1": p_all}).to_csv("p_layer1.csv", index=False)
    print("✅ Wrote p_layer1.csv")

    # approx coef importances (from one inner estimator)
    try:
        inner = clf_full.calibrated_classifiers_[0].base_estimator
        coefs = getattr(inner, "coef_", None)
        if coefs is not None:
            pd.DataFrame({"feature": X.columns, "coef": coefs.ravel()}).to_csv("coef_importance.csv", index=False)
            print("✅ Wrote coef_importance.csv")
    except Exception:
        pass

    # save model
    with open("model_layer1.pkl","wb") as f:
        cloudpickle.dump({"model": clf_full, "features": list(X.columns), "idcol": IDCOL, "label": LABEL}, f)
    print("✅ Saved model_layer1.pkl")

    # print final feature list used
    print("\nFeatures used:")
    for c in X.columns:
        print(" -", c)

if __name__ == "__main__":
    main()
