import json, os, re, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

LEAK_PAT = re.compile(r"(moca|diagnos|diag|pd[_ ]?dx|parkinson|label|target)", re.I)

def load_schema(path):
    if not os.path.exists(path):
        print(f"❌ schema not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def to_num(v):
    s = str(v).strip().lower()
    if s in {"1","yes","y","true","present"}: return 1.0
    if s in {"2","no","n","false","absent"}:  return 0.0
    try:
        return float(s)
    except:
        return np.nan

def metrics(y_true, p_hat):
    auroc = roc_auc_score(y_true, p_hat)
    auprc = average_precision_score(y_true, p_hat)
    brier = brier_score_loss(y_true, p_hat)
    prev  = float(np.mean(y_true))
    return dict(AUROC=auroc, AUPRC=auprc, Brier=brier, Prevalence=prev)

def leak_audit(feature_names):
    leaks = [f for f in feature_names if LEAK_PAT.search(f)]
    return leaks

def eval_layer(name, schema_path, probs_path, idcol_hint=None, label_hint=None):
    print(f"\n===== {name} =====")
    sch = load_schema(schema_path)
    if sch is None:
        print("⚠️ skipping (no schema)")
        return None

    data_path = sch.get("data", None) or sch.get("dataset") or sch.get("merged") or ""
    idcol = sch.get("idcol", idcol_hint)
    label = sch.get("label", label_hint)

    if not os.path.exists(data_path):
        print(f"❌ data file from schema not found: {data_path}")
        return None

    # load data + label
    df = pd.read_csv(data_path, low_memory=False)
    if label not in df.columns:
        print(f"❌ label '{label}' not in {data_path}")
        return None

    y = pd.to_numeric(df[label], errors="coerce")
    # load probs
    if not os.path.exists(probs_path):
        print(f"❌ probs file not found: {probs_path}")
        return None
    pr = pd.read_csv(probs_path)

    # align
    if idcol and idcol in df.columns and idcol in pr.columns:
        merged = df[[idcol, label]].merge(pr[[idcol] + [c for c in pr.columns if c != idcol]],
                                          on=idcol, how="inner")
        if merged.empty:
            print("❌ no overlap by ID between label and probs")
            return None
        y_true = pd.to_numeric(merged[label], errors="coerce").values
        # grab the first prob-like column
        pcols = [c for c in merged.columns if c not in (idcol, label) and merged[c].dtype != object]
        if not pcols:
            # try any non-id column
            pcols = [c for c in merged.columns if c not in (idcol, label)]
        p_hat = pd.to_numeric(merged[pcols[0]], errors="coerce").values
    else:
        # fall back to row order
        n = len(df)
        p = pr.select_dtypes(include=[np.number])
        if p.shape[0] != n:
            print(f"⚠️ could not align by ID, and length mismatch (labels={n}, probs={p.shape[0]}). Skipping.")
            return None
        y_true = y.values
        p_hat  = p.iloc[:,0].values

    # clean nan rows
    mask = ~np.isnan(y_true) & ~np.isnan(p_hat)
    y_true, p_hat = y_true[mask], p_hat[mask]
    if len(np.unique(y_true)) < 2:
        print("❌ label has a single class after cleaning; cannot score")
        return None

    m = metrics(y_true, p_hat)
    print(f"n={len(y_true)} | AUROC={m['AUROC']:.3f}  AUPRC={m['AUPRC']:.3f}  Brier={m['Brier']:.3f}  Prev={m['Prevalence']:.3f}")

    # feature list + leakage audit
    feats = []
    # schema may store features as dict or list
    fs = sch.get("features")
    if isinstance(fs, dict):
        feats = list(fs.values())
    elif isinstance(fs, list):
        feats = fs
    else:
        feats = []

    leaks = leak_audit([str(f) for f in feats])
    if leaks:
        print("⚠️ leakage-suspect feature names:", leaks[:20])
    else:
        print("✅ no obvious leakage terms in feature names")

    return {
        "layer": name,
        **m,
        "n_eval": int(len(y_true)),
        "features_count": int(len(feats)),
        "leakage_flags": ";".join(leaks[:20])
    }

def main():
    # defaults that match what you already created
    plans = [
        dict(name="Layer1_FoxInsight",
             schema="schema_lock_layer1.json",
             probs ="p_layer1.csv",
             id_hint=None, label_hint="CurrPDDiag"),

        dict(name="Layer2_PPMI",
             schema="schema_lock_layer2.json",
             probs ="p_layer2_ppmi.csv",
             id_hint="PATNO", label_hint="EarlyCogImp"),

        dict(name="Layer3_REDCap",
             schema="schema_lock_layer3.json",
             probs ="p_layer3_final.csv",
             id_hint=None, label_hint=None)
    ]

    rows = []
    for pl in plans:
        res = eval_layer(pl["name"], pl["schema"], pl["probs"], pl.get("id_hint"), pl.get("label_hint"))
        if res:
            rows.append(res)

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv("metrics_summary.csv", index=False)
        print("\n📄 wrote metrics_summary.csv")
        print(out.to_string(index=False))
    else:
        print("\n(no layers evaluated)")

if __name__ == "__main__":
    main()
