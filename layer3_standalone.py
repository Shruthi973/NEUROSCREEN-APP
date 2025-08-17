import argparse, json, os, re, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted
import cloudpickle


LEAK_PATTERNS = [
    r"diagnos", r"\bpd[_ ]?dx\b", r"parkinson", r"provider",
    r"mds[_-]?updrs.*(total|sum)", r"updrs.*(total|sum)",
    r"part[_ ]?iii\b", r"part[_ ]?iv\b", r"part[_ ]?3\b", r"part[_ ]?4\b",
    r"\bmoca.*total\b", r"\bmcatot\b", r"\bhy.*yahr\b", r"\bh.y\b"
]

# allow ONLY item-level Part I/II (not totals)
ALLOW_MDS_REGEX = r"^(mdsupdrs_1_|mdsupdrs_2_|mds[_-]?updrs[_-]?part[_ ]?(i|ii)_)"


def is_leak(col):
    name = str(col).lower()
    for p in LEAK_PATTERNS:
        if re.search(p, name):
            return True
    return False


def map_label_series(s):
    def _map(v):
        if pd.isna(v): return np.nan
        # numeric forms
        if isinstance(v, (int, float, np.integer, np.floating)):
            if v in [1, 1.0]: return 1
            if v in [0, 0.0]: return 0
        # string forms
        t = str(v).strip().lower()
        if t in {"1","yes","y","true","pd","parkinson","parkinson's","parkinsons","diagnosed"}: return 1
        if t in {"0","no","n","false","none","non-pd","not diagnosed","control"}: return 0
        if re.search(r"\b(yes|diagnos|parkinson)\b", t): return 1
        if re.search(r"\b(no|control)\b", t): return 0
        return np.nan
    return s.map(_map)


def pick_features(df, y, max_features):
    # separate types
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]

    # make MI-friendly table (factorize cats, fill NAs)
    X_mi = pd.DataFrame(index=df.index)
    for c in num_cols:
        X_mi[c] = df[c].copy()
    for c in cat_cols:
        X_mi[c] = pd.Categorical(df[c]).codes  # -1 for NaN

    # fill NA for MI
    for c in num_cols:
        X_mi[c] = X_mi[c].fillna(X_mi[c].median())
    for c in cat_cols:
        X_mi[c] = X_mi[c].replace(-1, X_mi[c].mode().iloc[0] if (X_mi[c] != -1).any() else 0)

    mi = mutual_info_classif(X_mi, y, discrete_features=[c in cat_cols for c in X_mi.columns], random_state=42)
    mi_series = pd.Series(mi, index=X_mi.columns).sort_values(ascending=False)

    # (1) take top-K by MI
    chosen = mi_series.head(max_features).index.tolist()

    # (2) ensure at least ONE allowed MDS item survives
    allowed_mds = [c for c in df.columns if re.search(ALLOW_MDS_REGEX, str(c).lower())]
    best_allowed = None
    if allowed_mds:
        # among allowed, take the one with highest MI
        allowed_sorted = sorted(allowed_mds, key=lambda c: mi_series.get(c, 0), reverse=True)
        if allowed_sorted:
            best_allowed = allowed_sorted[0]

    if best_allowed and best_allowed not in chosen:
        # replace the last one with the best allowed
        if len(chosen) >= max_features:
            chosen[-1] = best_allowed
        else:
            chosen.append(best_allowed)

    chosen = list(dict.fromkeys(chosen))  # dedupe, keep order
    return chosen, mi_series


def build_pipeline(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",    # stable with small N and L1
        class_weight="balanced",
        max_iter=2000,
        C=0.5
    )
    return Pipeline([("pre", pre), ("clf", model)])


def main():
    ap = argparse.ArgumentParser(description="Layer-3 (REDCap) leakage-guarded training")
    ap.add_argument("--data", required=True, help="CSV path (REDCap-like)")
    ap.add_argument("--idcol", default=None, help="Optional ID column to carry through")
    ap.add_argument("--label", default="screening_pd_dx", help="Label column (0/1 or yes/no)")
    ap.add_argument("--max_features", type=int, default=10, help="Top-K features by mutual information")
    ap.add_argument("--cv_splits", type=int, default=10, help="K in K-fold CV")
    ap.add_argument("--cv_repeats", type=int, default=5, help="Repeats for RepeatedStratifiedKFold")
    ap.add_argument("--min_nonnull", type=float, default=0.60, help="Drop columns with < this fraction present")
    ap.add_argument("--perm_test", type=int, default=0, help="If >0, run that many permutations for a sanity p-value")
    ap.add_argument("--out_probs", default="p_layer3.csv")
    ap.add_argument("--out_model", default="model_layer3.pkl")
    ap.add_argument("--out_feats", default="layer3_feature_importance.csv")
    ap.add_argument("--out_schema", default="schema_lock_layer3.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data, low_memory=False)

    if args.label not in df.columns:
        raise SystemExit(f"❌ label '{args.label}' not found in {args.data}")

    y = map_label_series(df[args.label])
    pos, neg = int((y==1).sum()), int((y==0).sum())
    print(f"Label summary → positives={pos} | negatives={neg} | missing={int(y.isna().sum())}")
    df = df[~y.isna()].copy()
    y = y[~y.isna()].astype(int)

    # drop label itself
    if args.label in df.columns:
        df = df.drop(columns=[args.label])

    # drop known leakage columns
    drop_cols = [c for c in df.columns if is_leak(c)]
    if drop_cols:
        print(f"⚠️ Dropping leakage-prone columns ({len(drop_cols)}), sample:", drop_cols[:12])
        df = df.drop(columns=drop_cols)

    # drop ultra-sparse
    keep_by_density = [c for c in df.columns if df[c].notna().mean() >= args.min_nonnull]
    dropped_sparse = [c for c in df.columns if c not in keep_by_density]
    if dropped_sparse:
        print(f"⚠️ Dropping sparse columns (<{args.min_nonnull:.0%} present): {len(dropped_sparse)}")
    df = df[keep_by_density]

    # feature selection (and force-keep best allowed MDS item)
    chosen, mi_series = pick_features(df, y, args.max_features)
    df = df[chosen].copy()

    # split by dtypes for the pipeline
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]

    pipe = build_pipeline(num_cols, cat_cols)

    # Repeated CV for robust estimate
    rskf = RepeatedStratifiedKFold(n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=42)
    aucs, prcs, briers = [], [], []
    any_bad_fold = False
    for fold, (tr, te) in enumerate(rskf.split(df, y), start=1):
        if len(np.unique(y.iloc[tr])) < 2 or len(np.unique(y.iloc[te])) < 2:
            any_bad_fold = True
            continue
        pipe.fit(df.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(df.iloc[te])[:,1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        prcs.append(average_precision_score(y.iloc[te], p))
        briers.append(brier_score_loss(y.iloc[te], p))

    if any_bad_fold:
        print("⚠️ Some folds had a single class; they were skipped (prevents fake AUC=1).")

    AUROC = float(np.mean(aucs)) if aucs else float("nan")
    AUPRC = float(np.mean(prcs)) if prcs else float("nan")
    BRIER = float(np.mean(briers)) if briers else float("nan")
    print("\nLayer-3 (REDCap, guarded)")
    print(f"  n={len(df)} | folds={args.cv_splits}x{args.cv_repeats}")
    print(f"  AUROC={AUROC:.3f}  AUPRC={AUPRC:.3f}  Brier={BRIER:.3f}  Prev={y.mean():.3f}")
    print(f"  Features used: {len(chosen)}")
    print("  Chosen:", chosen)

    # optional permutation sanity check
    if args.perm_test > 0 and not np.isnan(AUROC):
        rng = np.random.default_rng(0)
        perms = []
        for i in range(args.perm_test):
            yperm = pd.Series(rng.permutation(y.values), index=y.index)
            aucs_p = []
            for (tr, te) in rskf.split(df, yperm):
                if len(np.unique(yperm.iloc[tr])) < 2 or len(np.unique(yperm.iloc[te])) < 2:
                    continue
                pipe.fit(df.iloc[tr], yperm.iloc[tr])
                p = pipe.predict_proba(df.iloc[te])[:,1]
                aucs_p.append(roc_auc_score(yperm.iloc[te], p))
            if aucs_p:
                perms.append(np.mean(aucs_p))
        if perms:
            pval = (1 + sum(a >= AUROC for a in perms)) / (1 + len(perms))
            print("  Permuted AUROCs:", [round(x,3) for x in perms[:10]], "…")
            print(f"  Permutation p≈ {pval:.3f}")

    # Fit on all data for export
    pipe.fit(df, y)
    try:
        check_is_fitted(pipe)
    except Exception:
        raise SystemExit("❌ final model failed to fit; check label balance and features.")

    # export probabilities
    probs = pipe.predict_proba(df)[:,1]
    out_df = pd.DataFrame({"row_id": np.arange(len(df)), "p_layer3": probs})
    out_df.to_csv(args.out_probs, index=False)
    print(f"✅ wrote {args.out_probs}")

    # export feature contributions (coef magnitude on transformed space)
    # approximate: pull names from ColumnTransformer
    pre: ColumnTransformer = pipe.named_steps["pre"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    num_names = list(num_cols)
    cat_ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"] if cat_cols else None
    cat_names = (cat_ohe.get_feature_names_out(cat_cols).tolist() if cat_ohe else [])

    feat_names = num_names + cat_names
    coefs = np.ravel(clf.coef_)
    # guard mismatch lengths
    k = min(len(feat_names), len(coefs))
    fi = pd.DataFrame({"feature": feat_names[:k], "coef": coefs[:k], "abs_coef": np.abs(coefs[:k])})
    fi.sort_values("abs_coef", ascending=False).to_csv(args.out_feats, index=False)
    print(f"✅ wrote {args.out_feats}")

    # save model + schema
    schema = {
        "data": os.path.basename(args.data),
        "label": args.label,
        "min_nonnull": args.min_nonnull,
        "chosen_features": chosen,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "leak_guard": LEAK_PATTERNS,
        "allow_mds_regex": ALLOW_MDS_REGEX,
        "cv": {"splits": args.cv_splits, "repeats": args.cv_repeats}
    }
    with open(args.out_schema, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"✅ schema → {args.out_schema}")

    with open(args.out_model, "wb") as f:
        cloudpickle.dump({"pipeline": pipe, "schema": schema}, f)
    print(f"✅ saved {args.out_model}")


if __name__ == "__main__":
    main()
