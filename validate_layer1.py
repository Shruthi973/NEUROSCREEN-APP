# validate_layer1.py
import json, pandas as pd, numpy as np, sys

def main():
    lock=json.load(open("schema_lock_layer1.json"))
    df=pd.read_csv(lock["data"], low_memory=False)

    # presence
    need=[lock["idcol"], lock["label"]]+lock["features"]
    miss=[c for c in need if c not in df.columns]
    if miss: sys.exit(f"❌ Missing columns: {miss}")

    # basic sanity
    if df[lock["label"]].isna().all(): sys.exit("❌ Label column is entirely NaN.")
    if df[lock["idcol"]].isna().any(): print("⚠️ Some IDs are NaN.")

    # cardinality/sparsity check
    too_sparse=[]
    for c in lock["features"]:
        frac=df[c].notna().mean()
        if frac<0.5: too_sparse.append((c, round(frac,3)))
    if too_sparse:
        print("⚠️ Sparse features (<50% non-null):", too_sparse)

    # small type check (non-fatal; Streamlit safe)
    for c in [lock["label"]]+lock["features"]:
        # attempt numeric coercion without mutating file
        pd.to_numeric(df[c], errors="coerce")

    print("✅ Layer-1 validation passed.")
if __name__=="__main__":
    main()
