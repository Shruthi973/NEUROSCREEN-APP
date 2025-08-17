# merge_fox_v3_minimal.py
# Minimal, memory-safe per-ID merge for FoxInsight/PPMI-style drops.
# - Picks exactly 1 best column per feature (or via --override)
# - Aggregates each source to 1 row per ID (max after yes/no→1/0 coercion)
# - Skips huge/useless files
# - Writes: <OUT>.csv (+ <OUT>_feature_map.json)

import argparse, os, glob, re, sys, json
import pandas as pd
import numpy as np

CAND_ID_COLS = ["PATNO","fox_insight_id","participant_id","participantid","fox_id","subject_id","id"]
LABEL_COL_DEFAULT = None   # set a label column name if you have one (e.g., "CurrPDDiag"); or pass --label

# Features dictionary (keys = canonical names you want to keep in output)
FEATURE_KW = {
    # core risks (edit/trim as needed)
    "hyposmia": ["upsit","smell","olfactory","hyposmia","anosmia"],
    "rbd": ["rbdsq","rbd","rem sleep behavior"],
    "constipation": ["scopa","constip","gi"],
    "urinary_dysfunction": ["scopa","urinary","urin","incontinence","nocturia"],
    "dizziness": ["scopa","dizzi","cardio","cv"],
    "erectile_dysfunction": ["scopa","sex","erectile","ed"],
    # demographics
    "age": ["age","age at visit","years"],
    "gender": ["sex","gender"],
    # optional cognition complaint
    "memory_problems": ["pdaq","memory","recall"],
}

DEFAULT_SKIP = {"medicationspd.csv","code_list","data_diction","deprecated","high_interest"}  # lowercased substrings

def choose_id(cols):
    for c in CAND_ID_COLS:
        if c in cols: return c
    # try case-insensitive exact lookups
    low = {c.lower(): c for c in cols}
    for c in CAND_ID_COLS:
        if c.lower() in low: return low[c.lower()]
    return None

def score_column(col, kws):
    col = col.lower()
    sc = 0
    for kw in kws:
        if re.search(rf"\b{re.escape(kw)}\b", col): sc += 2
        elif kw in col: sc += 1
    return sc

def to_num(v):
    s = str(v).strip().lower()
    if s in {"1","yes","y","true","present"}: return 1.0
    if s in {"2","no","n","false","absent"}:  return 0.0
    if s in {"","nan","none","unknown","don’t know","dont know","dk"}: return np.nan
    try:
        return float(s)
    except:
        return np.nan

def read_head(path, n=5):
    return pd.read_csv(path, nrows=n, low_memory=False)

def chunk_reader(path, usecols, chunksize):
    # try fast pyarrow path, then safe fallback
    try:
        for ch in pd.read_csv(path, usecols=usecols, low_memory=False,
                              engine="pyarrow", dtype_backend="pyarrow",
                              chunksize=chunksize):
            yield ch
        return
    except Exception:
        pass
    for ch in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=chunksize):
        yield ch

def agg_chunk(df, idcol, feature_cols):
    df = df.copy()
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].map(to_num)
    keep_cols = [idcol] + [c for c in feature_cols if c in df.columns]
    if len(keep_cols) == 1:  # only ID is present
        return pd.DataFrame(columns=[idcol])
    agg_spec = {c: "max" for c in keep_cols if c != idcol}
    agg_spec[idcol] = "first"
    return df[keep_cols].groupby(idcol, as_index=False).agg(agg_spec)

def load_overrides(arg):
    """Supports:
       --override '{"hyposmia":"UPSIT_TOTAL",...}'
       --override overrides.json
       --override @overrides.json
    """
    if not arg:
        return {}
    if arg.startswith("@"):
        return json.load(open(arg[1:], "r"))
    # if it's a path without @
    if os.path.exists(arg):
        return json.load(open(arg, "r"))
    # else assume it's inline JSON
    return json.loads(arg)

def main():
    ap = argparse.ArgumentParser(description="Minimal per-ID merge into a single CSV")
    ap.add_argument("--src", required=True, help="Folder with CSVs (quote if spaces)")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--id", default=None, help="ID column name (e.g., PATNO)")
    ap.add_argument("--label", default=LABEL_COL_DEFAULT, help="Optional label column to include")
    ap.add_argument("--skip", default="", help="Comma-separated substrings to skip (case-insensitive)")
    ap.add_argument("--chunksize", type=int, default=200000)
    ap.add_argument("--override", default=None, help='JSON string or path or @path.json mapping {feature: exact_col}')
    args = ap.parse_args()

    src = args.src
    out = args.out
    user_skips = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    skip_set = DEFAULT_SKIP | user_skips

    csvs = sorted(glob.glob(os.path.join(src, "*.csv")))
    if not csvs:
        sys.exit(f"❌ No CSVs found in {src}")

    # Pick base file (try one that likely has the ID/label)
    base_path = None
    # prefer Demographics or General or first file
    prefs = ["demographics", "general", "about"]
    for pref in prefs:
        for p in csvs:
            if pref in os.path.basename(p).lower():
                base_path = p; break
        if base_path: break
    if base_path is None:
        base_path = csvs[0]

    base_head = read_head(base_path)
    idcol = args.id or choose_id(base_head.columns)
    if not idcol:
        # try scan all files for an ID
        for p in csvs:
            try:
                h = read_head(p)
                cand = choose_id(h.columns)
                if cand: 
                    idcol = cand; base_path = p; base_head = h; break
            except Exception:
                continue
    if not idcol:
        sys.exit("❌ No ID column found. Pass --id explicitly (e.g., --id PATNO).")

    label_col = args.label
    if label_col and label_col not in base_head.columns:
        # we'll try to pick it up later from other files; just warn
        print(f"⚠️ Label '{label_col}' not found in {os.path.basename(base_path)}; will try other files.")

    overrides = load_overrides(args.override)

    # 1) Choose best column per feature (overrides win)
    best = {k: (None, 0, None) for k in FEATURE_KW}  # (col, score, src)
    for p in csvs:
        name = os.path.basename(p); lname = name.lower()
        if any(s in lname for s in skip_set):
            # print(f"⏭  Skipping by rule: {name}")
            continue
        try:
            head = read_head(p, n=200)
        except Exception:
            continue
        cols = list(head.columns)
        for feat, kws in FEATURE_KW.items():
            if feat in overrides and overrides[feat] in cols:
                # hard override
                if best[feat][0] != overrides[feat]:
                    best[feat] = (overrides[feat], 999, name)
                continue
            # fallback: score by keywords
            local_best = (None, 0)
            for c in cols:
                sc = score_column(c, kws)
                if sc > local_best[1]:
                    local_best = (c, sc)
            if local_best[1] > best[feat][1]:
                best[feat] = (local_best[0], local_best[1], name)

    fmap = {feat: col for feat, (col, sc, src) in best.items() if col is not None}
    if not fmap:
        sys.exit("❌ No feature columns matched. Provide --override or adjust FEATURE_KW in the script.")

    wanted_cols = set([idcol] + list(fmap.values()) + ([label_col] if label_col else []))

    print("\n📋 Chosen columns:")
    for feat, (col, sc, src) in best.items():
        if col:
            tag = "OVERRIDE" if (feat in overrides and overrides[feat] == col) else f"score={sc}"
            print(f" - {feat:20s} → {col:35s} [{src}] ({tag})")
    print(f"✅ ID: {idcol}\n")

    # 2) Build minimal dataset starting from base
    base_use = [c for c in [idcol, label_col] if c and c in base_head.columns]
    base = pd.read_csv(base_path, usecols=base_use if base_use else [idcol], low_memory=False)
    base = base.drop_duplicates(subset=[idcol])
    merged = base

    for p in csvs:
        name = os.path.basename(p); lname = name.lower()
        if any(s in lname for s in skip_set):
            continue
        try:
            cols_now = read_head(p, n=1).columns
        except Exception:
            continue

        keep = [c for c in cols_now if c in wanted_cols]
        if idcol not in keep:
            if idcol in cols_now:
                keep = [idcol] + keep
            else:
                continue
        if set(keep) == {idcol}:
            continue

        print(f"… processing {name}")
        agg_df = None
        try:
            for chunk in chunk_reader(p, usecols=keep, chunksize=args.chunksize):
                part = agg_chunk(chunk, idcol, [c for c in keep if c != idcol])
                if part is None or part.empty: 
                    continue
                if agg_df is None:
                    agg_df = part
                else:
                    comb = pd.concat([agg_df, part], ignore_index=True)
                    agg_df = comb.groupby(idcol, as_index=False).max(numeric_only=True)
        except MemoryError:
            print(f"💥 MemoryError reading {name} — skipping"); continue
        except Exception as e:
            print(f"⚠️ Error reading {name}: {e} — skipping"); continue

        if agg_df is None or agg_df.empty: 
            continue
        dup_cols = [c for c in agg_df.columns if c in merged.columns and c != idcol]
        if dup_cols:
            agg_df = agg_df.drop(columns=dup_cols, errors="ignore")
        try:
            merged = merged.merge(agg_df, on=idcol, how="left")
        except MemoryError:
            print(f"💥 MemoryError merging {name} — skipping merge"); continue
        except Exception as e:
            print(f"⚠️ Merge failed for {name}: {e}"); continue

    # Keep only ID, label, and chosen feature cols (in FEATURE_KW order)
    ordered_cols = [idcol]
    if label_col and label_col in merged.columns:
        ordered_cols.append(label_col)
    ordered_cols += [fmap[k] for k in FEATURE_KW if k in fmap and fmap[k] in merged.columns]
    merged = merged.loc[:, ordered_cols].copy()
    merged.to_csv(out, index=False)

    # Save the mapping used (handy for Streamlit + reproducibility)
    with open(out.replace(".csv", "_feature_map.json"), "w") as f:
        json.dump({k: v for k, v in fmap.items()}, f, indent=2)

    print(f"\n🎉 Wrote {out} with shape {merged.shape} and {len(ordered_cols)} columns.")
    print(f"🗺️  Wrote {out.replace('.csv','_feature_map.json')}")

if __name__ == "__main__":
    main()
