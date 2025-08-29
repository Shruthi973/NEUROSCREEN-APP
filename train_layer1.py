#!/usr/bin/env python3
# train_layer1.py
# ------------------------------------------------------------------
# Layer 1 (Fox Insight) — Full analysis with 70/10/20 split + p-values
#
# Conventions:
#   IDCOL  = "fox_insight_id"
#   LABEL  = "CurrPDDiag"   (must be binary 0/1)
#   CAND_FEATS = [
#       ImpactMoveHyposmia, ImpactMoveConstipation, Anxiety, ImpactMoveUrinary,
#       ImpactMoveTremor, ImpactThinkBodyFatigue, ImpactThinkDizzy,
#       ImpactMoveImbalance, ImpactThinkMemory, MoveSaliva, age, Sex
#   ]
# ------------------------------------------------------------------

import argparse, json, joblib, numpy as np, pandas as pd, sys
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    brier_score_loss, log_loss, accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

IDCOL = "fox_insight_id"
LABEL = "CurrPDDiag"

CAND_FEATS = [
    "ImpactMoveHyposmia","ImpactMoveConstipation","Anxiety","ImpactMoveUrinary",
    "ImpactMoveTremor","ImpactThinkBodyFatigue","ImpactThinkDizzy",
    "ImpactMoveImbalance","ImpactThinkMemory","MoveSaliva","age","Sex",
]

def youden_threshold(y_true, p):
    fpr, tpr, thr = roc_curve(y_true, p)
    j = tpr - fpr
    k = int(np.argmax(j)) if len(j) else 0
    return float(thr[k]) if len(thr) else 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Layer-1 training CSV (wide)")
    ap.add_argument("--outdir", default="layer1_fox_out", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.70, help="Train fraction")
    ap.add_argument("--valid_frac", type=float, default=0.10, help="Valid fraction (test is rest)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Load ----------------
    df = pd.read_csv(args.csv, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"[read] {args.csv}  shape={df.shape}")

    # ---------------- Label ----------------
    if LABEL not in df.columns:
        sys.exit("❌ L1: CurrPDDiag not found.")
    y = pd.to_numeric(df[LABEL], errors="coerce")
    mask = y.isin([0, 1])
    if not mask.any():
        sys.exit("❌ L1: No valid binary labels (0/1) in CurrPDDiag.")
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int)
    if y.nunique() != 2:
        sys.exit(f"❌ L1: CurrPDDiag must be binary 0/1. Got: {sorted(y.unique())}")

    # ---------------- Features ----------------
    feats_present = [c for c in CAND_FEATS if c in df.columns and df[c].notna().any()]
    if not feats_present:
        sys.exit("❌ No usable L1 features found from CAND_FEATS.")
    Xraw = df[feats_present].copy()

    # Map Sex explicitly (includes 1/2 → 1/0)
    if "Sex" in Xraw.columns:
        Xraw["Sex"] = (
            Xraw["Sex"].astype(str).str.upper().str.strip()
            .map({
                "MALE": 1, "M": 1, "1": 1,            # male codes
                "FEMALE": 0, "F": 0, "0": 0, "2": 0   # female codes
            })
        )
        Xraw["Sex"] = pd.to_numeric(Xraw["Sex"], errors="coerce")

    # Other features must already be numeric (no blind coercion)
    non_sex = [c for c in feats_present if c != "Sex"]
    non_numeric = [c for c in non_sex if not pd.api.types.is_numeric_dtype(Xraw[c])]
    if non_numeric:
        raise ValueError(f"Expected numeric features but found non-numeric: {non_numeric}")

    # Final list; include Sex only if it has at least one value
    num_cols = non_sex + (["Sex"] if ("Sex" in Xraw.columns and Xraw["Sex"].notna().any()) else [])
    if not num_cols:
        sys.exit("❌ No numeric columns after strict checks; cannot train.")
    X = Xraw[num_cols].copy()

    # ---------------- IDs ----------------
    if IDCOL in df.columns:
        ids = df[IDCOL].astype(str).str.strip().values
    else:
        print(f"[warn] {IDCOL} not found; generating synthetic ids.")
        ids = np.arange(len(df)).astype(str)

    # ---------------- Split 70/10/20 ----------------
    test_size = 1.0 - args.train_frac
    valid_rel = args.valid_frac / max(test_size, 1e-9)
    X_train, X_hold, y_train, y_hold, id_tr, id_hold = train_test_split(
        X, y.values, ids, test_size=test_size, random_state=args.seed, stratify=y.values
    )
    X_valid, X_test, y_valid, y_test, id_va, id_te = train_test_split(
        X_hold, y_hold, id_hold, test_size=(1.0 - valid_rel), random_state=args.seed+1, stratify=y_hold
    )
    print(f"[split] train={len(X_train)}  valid={len(X_valid)}  test={len(X_test)}")
    print(f"[balance] train PD%={y_train.mean():.3f}  valid PD%={y_valid.mean():.3f}  test PD%={y_test.mean():.3f}")

    # ---------------- Pipeline ----------------
    numeric = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
    ])
    pre = ColumnTransformer(transformers=[("num", numeric, num_cols)], remainder="drop")
    clf = LogisticRegression(solver="liblinear", max_iter=1000, class_weight=None, random_state=args.seed)
    pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])

    # ---------------- Fit ----------------
    pipe.fit(X_train, y_train)

    # ---------------- Validation (threshold) ----------------
    p_va = pipe.predict_proba(X_valid)[:, 1]
    auc_v = roc_auc_score(y_valid, p_va)
    ap_v  = average_precision_score(y_valid, p_va)
    thr   = youden_threshold(y_valid, p_va)
    pred_v = (p_va >= thr).astype(int)
    acc_v = accuracy_score(y_valid, pred_v)
    sens_v = ((pred_v==1)&(y_valid==1)).sum()/max(1,(y_valid==1).sum())
    spec_v = ((pred_v==0)&(y_valid==0)).sum()/max(1,(y_valid==0).sum())
    brier_v = brier_score_loss(y_valid, p_va)
    ll_v    = log_loss(y_valid, np.clip(p_va, 1e-9, 1-1e-9))
    print(f"VAL : AUC={auc_v:.3f}  AP={ap_v:.3f}  thr*={thr:.3f}  sens={sens_v:.3f}  spec={spec_v:.3f}  acc={acc_v:.3f}")

    # ---------------- Train quick metrics ----------------
    p_tr = pipe.predict_proba(X_train)[:, 1]
    auc_tr = roc_auc_score(y_train, p_tr)
    ap_tr  = average_precision_score(y_train, p_tr)
    brier_tr = brier_score_loss(y_train, p_tr)
    ll_tr    = log_loss(y_train, np.clip(p_tr, 1e-9, 1-1e-9))
    pred_tr  = (p_tr >= thr).astype(int)
    acc_tr   = accuracy_score(y_train, pred_tr)
    sens_tr  = ((pred_tr==1)&(y_train==1)).sum()/max(1,(y_train==1).sum())
    spec_tr  = ((pred_tr==0)&(y_train==0)).sum()/max(1,(y_train==0).sum())
    print(f"TRN : AUC={auc_tr:.3f}  AP={ap_tr:.3f}  acc={acc_tr:.3f}")

    # ---------------- Test at VALID threshold ----------------
    p_te = pipe.predict_proba(X_test)[:, 1]
    auc_te = roc_auc_score(y_test, p_te)
    ap_te  = average_precision_score(y_test, p_te)
    brier_te = brier_score_loss(y_test, p_te)
    ll_te    = log_loss(y_test, np.clip(p_te, 1e-9, 1-1e-9))
    pred_te = (p_te >= thr).astype(int)
    acc_te  = accuracy_score(y_test, pred_te)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_te).ravel()
    sens_te = tp / max(1, (tp + fn))
    spec_te = tn / max(1, (tn + fp))
    print(f"TEST: AUC={auc_te:.3f}  AP={ap_te:.3f}  sens={sens_te:.3f}  spec={spec_te:.3f}  acc={acc_te:.3f}")
    print(f"TEST: Confusion: TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # ---------------- Save model dict ----------------
    model = {
        "pipeline": pipe,
        "features": num_cols,
        "threshold": float(thr),
        "label": LABEL,
        "idcol": IDCOL,
        "metrics": {
            "train_auc": float(auc_tr), "train_ap": float(ap_tr),
            "train_acc": float(acc_tr), "train_sens": float(sens_tr), "train_spec": float(spec_tr),
            "train_brier": float(brier_tr), "train_logloss": float(ll_tr),
            "val_auc": float(auc_v), "val_ap": float(ap_v),
            "val_thr": float(thr), "val_acc": float(acc_v),
            "val_sens": float(sens_v), "val_spec": float(spec_v),
            "val_brier": float(brier_v), "val_logloss": float(ll_v),
            "test_auc": float(auc_te), "test_ap": float(ap_te),
            "test_acc": float(acc_te), "test_sens": float(sens_te), "test_spec": float(spec_te),
            "test_brier": float(brier_te), "test_logloss": float(ll_te),
            "test_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }
    }
    model_path = outdir / "model_layer1.pkl"
    joblib.dump(model, model_path); print(f"💾 Saved {model_path}")

    # ---------------- Summary outputs ----------------
    pd.DataFrame({"feature": num_cols}).to_csv(outdir / "feature_list.csv", index=False)
    metrics_df = pd.DataFrame([
        {"split":"train","auc":auc_tr,"ap":ap_tr,"acc":acc_tr,"sens":sens_tr,"spec":spec_tr,"brier":brier_tr,"logloss":ll_tr,"threshold":thr},
        {"split":"valid","auc":auc_v,"ap":ap_v,"acc":acc_v,"sens":sens_v,"spec":spec_v,"brier":brier_v,"logloss":ll_v,"threshold":thr},
        {"split":"test", "auc":auc_te,"ap":ap_te,"acc":acc_te,"sens":sens_te,"spec":spec_te,"brier":brier_te,"logloss":ll_te,"threshold":thr,
         "tn":tn,"fp":fp,"fn":fn,"tp":tp}
    ])
    metrics_df.to_csv(outdir / "metrics_summary.csv", index=False)
    with open(outdir / "threshold.json", "w") as f:
        json.dump({"threshold": float(thr), "picked_on":"valid"}, f, indent=2)
    print(f"📝 Wrote {outdir/'feature_list.csv'}, {outdir/'metrics_summary.csv'}, {outdir/'threshold.json'}")

    # ---------------- Per-row probabilities (for ensemble) ----------------
    def _mk_probs(ids_, probs, split_name):
        return pd.DataFrame({"id": ids_, "split": split_name, "p_layer1": probs})
    probs_all = pd.concat([
        _mk_probs(id_tr, pipe.predict_proba(X_train)[:,1], "train"),
        _mk_probs(id_va, p_va, "valid"),
        _mk_probs(id_te, p_te, "test"),
    ], ignore_index=True)
    probs_all.to_csv(outdir / "probs_layer1.csv", index=False)
    print(f"📝 Wrote {outdir/'probs_layer1.csv'}")

    # ---------------- Coefficient p-values (robust) ----------------
    try:
        X_train_mat = pipe.named_steps["prep"].transform(X_train)
        feat_names = list(num_cols)
        Z = pd.DataFrame(X_train_mat, columns=feat_names)

        # 1) Drop zero-variance
        keep = Z.columns[Z.var() > 0]
        dropped_zero_var = [c for c in Z.columns if c not in keep]
        Z = Z[keep]

        # 2) Drop duplicate columns
        to_drop = set()
        arr = Z.values
        for i, c1 in enumerate(Z.columns):
            if c1 in to_drop: 
                continue
            for j, c2 in enumerate(Z.columns):
                if j > i and c2 not in to_drop and np.allclose(arr[:, i], arr[:, j]):
                    to_drop.add(c2)
        if to_drop:
            Z = Z.drop(columns=list(to_drop))

        # 3) Drop perfectly (near-perfectly) correlated columns
        corr = Z.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_corr = set(
            c for r in upper.index for c in upper.columns
            if pd.notna(upper.loc[r, c]) and upper.loc[r, c] >= 0.999
        )
        if drop_corr:
            Z = Z.drop(columns=list(drop_corr))

        # 4) Fit Logit; fallback to GLM Binomial with robust SE
        X_sm = sm.add_constant(Z, has_constant="add")
        try:
            res = sm.Logit(y_train, X_sm).fit(disp=False)
            coefs, bse, pvals = res.params, res.bse, res.pvalues
        except Exception:
            glm = sm.GLM(y_train, X_sm, family=sm.families.Binomial())
            res = glm.fit(cov_type="HC0")
            coefs, bse, pvals = res.params, res.bse, res.pvalues

        OR    = np.exp(coefs)
        CI_lo = np.exp(coefs - 1.96*bse)
        CI_hi = np.exp(coefs + 1.96*bse)

        coef_df = pd.DataFrame({
            "term": X_sm.columns,
            "coef": coefs,
            "std_err": bse,
            "z": coefs / bse,
            "p_value": pvals,
            "odds_ratio": OR,
            "or_95ci_lo": CI_lo,
            "or_95ci_hi": CI_hi,
        })
        coef_df.to_csv(outdir / "lr_coeff_pvalues.csv", index=False)

        with open(outdir / "pvalue_dropped_columns.txt", "w") as f:
            if dropped_zero_var:
                f.write(f"zero_variance_dropped: {dropped_zero_var}\n")
            if to_drop:
                f.write(f"duplicates_dropped: {sorted(list(to_drop))}\n")
            if drop_corr:
                f.write(f"perfect_corr_dropped: {sorted(list(drop_corr))}\n")

        print("📝 Wrote", outdir / "lr_coeff_pvalues.csv")
    except Exception as e:
        print(f"[warn] statsmodels p-values not computed even after cleanup: {e}")

if __name__ == "__main__":
    main()
