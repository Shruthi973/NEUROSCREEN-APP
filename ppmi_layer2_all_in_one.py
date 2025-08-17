# ppmi_layer2_all_in_one.py
# One-shot Layer-2 (PPMI): optional merge -> label -> flexible feature map -> train -> export
import argparse, json, os, subprocess, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import cloudpickle

def run(cmd):
    print("▶", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(f"❌ command failed: {' '.join(cmd)}")

def to_num(v):
    s = str(v).strip().lower()
    if s in {"1","yes","y","true","present"}: return 1.0
    if s in {"2","no","n","false","absent"}:  return 0.0
    if s in {"","nan","none","unknown","don’t know","dont know","dk"}: return np.nan
    try: return float(s)
    except: return np.nan

def build_model(n_features: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    base = LogisticRegression(
        solver="liblinear",       # stable on small / sparse feature sets
        class_weight="balanced",
        max_iter=5000
    )
    return CalibratedClassifierCV(base, cv=5, method="isotonic")


def main():
    ap = argparse.ArgumentParser(description="Layer-2 PPMI pipeline (flexible features, streamlit-safe)")
    # Optional merge
    ap.add_argument("--src", default=None, help="Folder with raw PPMI CSVs (if given, runs merge_fox_v3_minimal.py)")
    ap.add_argument("--merge_out", default="PPMI_clinical.csv", help="Merged clinical CSV output")
    ap.add_argument("--merge_skip", default="", help="Comma-separated substrings to skip during merge")
    ap.add_argument("--merge_override", default=None, help="Path to JSON overrides for merge (passed through)")
    ap.add_argument("--idcol", default="PATNO", help="Participant ID column in merged CSV")

    # Label
    ap.add_argument("--moca_col", default="MCATOT", help="MoCA total column name")
    ap.add_argument("--moca_cut", type=float, default=26.0, help="Impairment cutoff: score < cut => 1")
    ap.add_argument("--label_name", default="EarlyCogImp", help="Name for generated label")
    ap.add_argument("--labeled_out", default="PPMI_clinical_labeled.csv", help="Output labeled CSV")

    # Feature mapping (flexible)
    ap.add_argument("--map", required=True, help="JSON with {'idcol','label','features':{canon: exact_col}}")
    ap.add_argument("--min_nonnull", type=float, default=0.10, help="Min fraction non-null to keep a feature (default 0.10)")

    # Outputs
    ap.add_argument("--model_out", default="model_layer2_ppmi.pkl")
    ap.add_argument("--probs_out", default="p_layer2_ppmi.csv")
    ap.add_argument("--schema_out", default="schema_lock_layer2.json")

    args = ap.parse_args()

    # 0) Optional merge
    if args.src:
        if not os.path.exists("merge_fox_v3_minimal.py"):
            raise SystemExit("❌ merge_fox_v3_minimal.py not found.")
        cmd = ["python", "merge_fox_v3_minimal.py", "--src", args.src, "--out", args.merge_out]
        if args.merge_skip:
            cmd += ["--skip", args.merge_skip]
        if args.merge_override:
            # allow raw json or @file
            pref = "@" if not args.merge_override.startswith("@") else ""
            cmd += ["--override", f"{pref}{args.merge_override}"]
        run(cmd)
    else:
        if not os.path.exists(args.merge_out):
            raise SystemExit(f"❌ {args.merge_out} not found and --src not provided.")

    # 1) Label
    df = pd.read_csv(args.merge_out, low_memory=False)
    if args.moca_col not in df.columns:
        raise SystemExit(f"❌ MoCA column '{args.moca_col}' not in {args.merge_out}")
    if args.idcol not in df.columns:
        raise SystemExit(f"❌ ID column '{args.idcol}' not in {args.merge_out}")

    moca = pd.to_numeric(df[args.moca_col], errors="coerce")
    df[args.label_name] = (moca < args.moca_cut).astype(float)
    df.to_csv(args.labeled_out, index=False)
    print(f"✅ Labeled → {args.labeled_out}  (label='{args.label_name}', cut={args.moca_cut})")

    # 2) Load map; flexibly intersect with available columns
    cfg = json.load(open(args.map))
    idcol = cfg.get("idcol", args.idcol)
    label = cfg.get("label", args.label_name)
    fmap = cfg["features"]

    df2 = pd.read_csv(args.labeled_out, low_memory=False)

    # Use only features whose mapped columns actually exist
    present = {k: v for k, v in fmap.items() if v in df2.columns}
    missing = {k: v for k, v in fmap.items() if v not in df2.columns}
    if missing:
        print("ℹ️ Skipping missing mapped columns:", missing)

    need_cols = [idcol, label] + list(present.values())
    miss_core = [c for c in [idcol, label] if c not in df2.columns]
    if miss_core:
        raise SystemExit(f"❌ Missing core columns {miss_core} in {args.labeled_out}")

    # Schema lock (for the columns we will actually use)
    schema = {
        "data": args.labeled_out,
        "idcol": idcol,
        "label": label,
        "features": present,
        "dtypes": {c: str(df2[c].dtype) for c in [idcol, label] + list(present.values()) if c in df2.columns}
    }
    json.dump(schema, open(args.schema_out, "w"), indent=2)
    print(f"✅ Schema lock → {args.schema_out}")

    # 3) Build X/y with flexible keep rules
    X = df2[list(present.values())].copy()
    for c in X.columns: X[c] = X[c].map(to_num)
    # keep features with enough data
    keep_mask = X.notna().mean() >= args.min_nonnull
    keep = list(X.columns[keep_mask])
    if not keep:
        raise SystemExit("❌ After non-null filtering, no features remain. Lower --min_nonnull or add more columns.")
    dropped = [c for c in X.columns if c not in keep]
    if dropped:
        print("⚠️ Dropping sparse features:", dropped)
    X = X[keep].copy()

    y = pd.to_numeric(df2[label], errors="coerce")
    ids = df2[idcol].values
    rowmask = (~y.isna()) & (X.notna().sum(axis=1) > 0)
    X, y, ids = X[rowmask], y[rowmask], ids[rowmask]
    X = X.fillna(X.median(numeric_only=True))

    # Split
    strat_ok = y.nunique() == 2 and all((y.value_counts() >= 2))
    test_size = 0.2 if len(y) >= 50 else 0.3
    rs = 42
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=rs,
        stratify=y if strat_ok else None
    )

    # Model (auto-select solver)
    clf = build_model(X.shape[1])
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]

    print("\nLayer-2 PPMI (all-in-one)")
    print("  AUROC:", round(roc_auc_score(yte, p), 3))
    print("  AUPRC:", round(average_precision_score(yte, p), 3))
    print("  Brier:", round(brier_score_loss(yte, p), 3))
    print(f"  Features used: {X.shape[1]}  ({', '.join(keep)})")

    # Fit on all data and export
    clf_full = build_model(X.shape[1])
    clf_full.fit(X, y)
    p_all = clf_full.predict_proba(X)[:, 1]
    pd.DataFrame({idcol: ids, "p_layer2": p_all}).to_csv(args.probs_out, index=False)
    print(f"✅ Wrote {args.probs_out}")

    with open(args.model_out, "wb") as f:
        cloudpickle.dump({"model": clf_full, "features": list(X.columns),
                          "idcol": idcol, "label": label}, f)
    print(f"✅ Saved {args.model_out}")

if __name__ == "__main__":
    main()
