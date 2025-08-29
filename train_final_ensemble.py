# train_final_ensemble.py
# ------------------------------------------------------------
# SIMPLE ENSEMBLE for PD risk (no calibration, no CV).
# - Tries FEATURES MODE (score 3 models from a single intake CSV).
# - If that fails or yields no usable probs, FALLS BACK to PROBS MODE
#   (blend per-layer probability CSVs).
#
# Fixed default weights: L1=0.20, L2=0.40, L3=0.40 (renormalized per-row
# when some layers are missing). Single risk categorization:
#   low < 0.33, 0.33<=medium<0.66, high >= 0.66
#
# Layer outputs this script expects by default:
#   L1 model:   layer1_fox_out/model_layer1.pkl
#   L1 probs:   layer1_fox_out/probs_layer1.csv  (id, split, p_layer1)
#   L2 model:   layer2_ppmi_out/model_layer2_ppmi.pkl
#   L2 probs:   layer2_ppmi_out/p_layer2_ppmi.csv (PATNO, p_layer2)
#   L3 model:   layer3_shortened_out/model_layer3_shortened.pkl
#   L3 probs:   layer3_shortened_out/p_layer3.csv (id, p_layer3)
#
# Outputs:
#   final_ensemble/weights.json          (base weights)
#   final_ensemble/config.json           (mode & params)
#   final_ensemble/combined_probs.csv    (id, p_layer*, used_layers, p_final, risk_category, advice)
# ------------------------------------------------------------

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ---------- Defaults ----------
DEFAULT_WEIGHTS = {"layer1": 0.20, "layer2": 0.40, "layer3": 0.40}
DEFAULT_LOW_THR = 0.33
DEFAULT_HIGH_THR = 0.66

ADVICE_MAP = {
    "low":   "Low risk: maintain healthy lifestyle and routine check-ups.",
    "medium":"Moderate risk: consider neurological evaluation within 3–6 months.",
    "high":  "High risk: prompt neurological consultation and confirmatory testing is recommended."
}

# Fallback feature lists (used only if we can’t read feature files from disk)
FALLBACK_L1 = [
    "ImpactMoveHyposmia","ImpactMoveConstipation","Anxiety","ImpactMoveUrinary",
    "ImpactMoveTremor","ImpactThinkBodyFatigue","ImpactThinkDizzy","ImpactMoveImbalance",
    "ImpactThinkMemory","MoveSaliva","age","Sex"
]
FALLBACK_L2 = [
    "AGE_AT_VISIT","SEX","DIFFRECALL","MCAVIGIL","NP1URIN_OL","RBD_SOURCE_OL","UPSIT_PRESENT",
    "RBD_PRESENT","URIN_PRESENT","MEM_PRESENT","PQUEST_SOURCE_OL","TRBBUTTN_OL","WRTSMLR_OL",
    "DFCLTYTYPE_OL","VOICSFTR_OL","FTSTUCK_OL","LSSXPRSS_OL","ARMLGSHK_OL",
    "TRBUPCHR_OL","SHUFFLE_OL","MVSLOW_OL","POORBAL_OL"
]
FALLBACK_L3 = [
    "screening_age","sex","anxiety_score","conscious_movement_score",
    "screening_walk","cdte","cogdt","dt_tug_time"
]

# ---------- Utils ----------
def _norm_id_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)

def _normalize_weights(weights: dict, present_layers: list[str]) -> dict:
    # Renormalize only across layers that provided a prob for the row
    w = {k: (weights.get(k, 0.0) if k in present_layers else 0.0) for k in ["layer1","layer2","layer3"]}
    s = float(sum(w.values()))
    if s <= 0:
        eq = 1.0 / max(len(present_layers), 1)
        return {k: (eq if k in present_layers else 0.0) for k in w}
    return {k: (v / s) for k, v in w.items()}

def _categorize(p, low_thr=DEFAULT_LOW_THR, high_thr=DEFAULT_HIGH_THR):
    if pd.isna(p): return "unknown"
    if p < low_thr: return "low"
    if p < high_thr: return "medium"
    return "high"

def _advice_for(cat: str) -> str:
    return ADVICE_MAP.get(cat, "No advice available.")

def _load_model(path: str | Path):
    return joblib.load(path)

def _read_features_csv(path: Path, colname="feature") -> list | None:
    if path.exists():
        try:
            return pd.read_csv(path)[colname].astype(str).tolist()
        except Exception:
            return None
    return None

def _get_features(model_path: Path, bundle_or_pipe) -> list | None:
    # 1) Bundle with features
    if isinstance(bundle_or_pipe, dict) and "features" in bundle_or_pipe:
        return list(bundle_or_pipe["features"])
    # 2) Nearby feature files by convention
    for cand in ["feature_list.csv","ppmi_feature_list.csv","layer3_feature_list.csv"]:
        f = model_path.parent / cand
        feats = _read_features_csv(f)
        if feats: return feats
    # 3) Try sklearn attribute (may or may not exist)
    try:
        names = list(getattr(bundle_or_pipe, "feature_names_in_", []))
        if names: return names
    except Exception:
        pass
    return None

def _row_to_X(row: pd.Series, feats: list[str]) -> pd.DataFrame:
    # Build a one-row DF with exactly these columns, pulling from row or NaN
    return pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])

def _read_prob_csv(path, id_col, p_col) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"[warn] missing probs file: {path}")
        return pd.DataFrame(columns=["id", p_col])
    df = pd.read_csv(path)
    if id_col not in df.columns or p_col not in df.columns:
        raise ValueError(f"{path} must have columns [{id_col}, {p_col}]")
    out = df[[id_col, p_col]].copy()
    out.rename(columns={id_col: "id"}, inplace=True)
    out["id"] = _norm_id_series(out["id"])
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce")
    return out.dropna(subset=[p_col]).reset_index(drop=True)

# ---------- Features Mode ----------
def run_features_mode(args) -> pd.DataFrame | None:
    df = pd.read_csv(args.input_csv, low_memory=False)
    df.columns = df.columns.str.strip()
    if args.id_col not in df.columns:
        df[args.id_col] = np.arange(len(df)).astype(str)

    # Load models (tolerate missing)
    m1 = _load_model(args.layer1_model) if Path(args.layer1_model).exists() else None
    m2 = _load_model(args.layer2_model) if Path(args.layer2_model).exists() else None
    m3 = _load_model(args.layer3_model) if Path(args.layer3_model).exists() else None

    # Extract pipelines from bundles if needed
    pipe1 = m1["pipeline"] if isinstance(m1, dict) and "pipeline" in m1 else m1
    pipe2 = m2["pipeline"] if isinstance(m2, dict) and "pipeline" in m2 else m2
    pipe3 = m3["pipeline"] if isinstance(m3, dict) and "pipeline" in m3 else m3

    # Features per model (fallback to hard-coded lists if none found)
    feats1 = _get_features(Path(args.layer1_model), m1) if m1 is not None else None
    feats2 = _get_features(Path(args.layer2_model), m2) if m2 is not None else None
    feats3 = _get_features(Path(args.layer3_model), m3) if m3 is not None else None
    if m1 is not None and feats1 is None: feats1 = FALLBACK_L1
    if m2 is not None and feats2 is None: feats2 = FALLBACK_L2
    if m3 is not None and feats3 is None: feats3 = FALLBACK_L3

    rows = []
    for _, row in df.iterrows():
        rid = _norm_id_series(pd.Series([row[args.id_col]])).iloc[0]
        rec = {"id": rid}

        try:
            if pipe1 is not None and feats1:
                X1 = _row_to_X(row, feats1)
                rec["p_layer1"] = float(pipe1.predict_proba(X1)[:,1][0])
        except Exception as e:
            print(f"[warn] L1 scoring failed for id={rid}: {e}")

        try:
            if pipe2 is not None and feats2:
                X2 = _row_to_X(row, feats2)
                rec["p_layer2"] = float(pipe2.predict_proba(X2)[:,1][0])
        except Exception as e:
            print(f"[warn] L2 scoring failed for id={rid}: {e}")

        try:
            if pipe3 is not None and feats3:
                X3 = _row_to_X(row, feats3)
                rec["p_layer3"] = float(pipe3.predict_proba(X3)[:,1][0])
        except Exception as e:
            print(f"[warn] L3 scoring failed for id={rid}: {e}")

        present = [L for L in ["layer1","layer2","layer3"] if pd.notna(rec.get(f"p_{L}"))]
        if present:
            w = _normalize_weights({"layer1": args.w1, "layer2": args.w2, "layer3": args.w3}, present)
            rec["used_layers"] = "|".join(present)
            rec["p_final"] = float(
                (rec.get("p_layer1", 0.0) * w.get("layer1", 0.0)) +
                (rec.get("p_layer2", 0.0) * w.get("layer2", 0.0)) +
                (rec.get("p_layer3", 0.0) * w.get("layer3", 0.0))
            )
        else:
            rec["used_layers"] = ""
            rec["p_final"] = np.nan

        rows.append(rec)

    combined = pd.DataFrame(rows)
    # If all rows have NaN p_final, return None to trigger fallback
    if combined["p_final"].isna().all():
        return None

    combined["risk_category"] = combined["p_final"].apply(lambda x: _categorize(x, args.low_thr, args.high_thr))
    combined["advice"] = combined["risk_category"].apply(_advice_for)
    return combined

# ---------- Probs Mode ----------
def run_probs_mode(args) -> pd.DataFrame | None:
    frames = []

    # L1: layer1_fox_out/probs_layer1.csv  (id, split, p_layer1)
    if args.p1 and Path(args.p1).exists():
        df1 = pd.read_csv(args.p1)
        if args.id1 not in df1.columns or args.p1col not in df1.columns:
            raise SystemExit(f"{args.p1} must have columns [{args.id1}, {args.p1col}]")
        df1 = df1[[args.id1, args.p1col]].copy()
        df1.rename(columns={args.id1:"id"}, inplace=True)
        df1["id"] = _norm_id_series(df1["id"])
        df1[args.p1col] = pd.to_numeric(df1[args.p1col], errors="coerce")
        df1 = df1.dropna(subset=[args.p1col]).rename(columns={args.p1col:"p_layer1"})
        frames.append(df1)

    # L2: layer2_ppmi_out/p_layer2_ppmi.csv  (PATNO, p_layer2)
    if args.p2 and Path(args.p2).exists():
        df2 = _read_prob_csv(args.p2, args.id2, args.p2col).rename(columns={args.p2col:"p_layer2"})
        frames.append(df2)

    # L3: layer3_shortened_out/p_layer3.csv  (id, p_layer3)
    if args.p3 and Path(args.p3).exists():
        df3 = _read_prob_csv(args.p3, args.id3, args.p3col).rename(columns={args.p3col:"p_layer3"})
        frames.append(df3)

    if not frames:
        return None

    from functools import reduce
    probs = reduce(lambda L,R: pd.merge(L,R,on="id", how="outer"), frames)

    def _blend_row(r):
        present = []
        if pd.notna(r.get("p_layer1")): present.append("layer1")
        if pd.notna(r.get("p_layer2")): present.append("layer2")
        if pd.notna(r.get("p_layer3")): present.append("layer3")
        if not present:
            return np.nan, ""
        w = _normalize_weights({"layer1": args.w1, "layer2": args.w2, "layer3": args.w3}, present)
        p_final = float(
            (r.get("p_layer1", 0.0) * w.get("layer1", 0.0)) +
            (r.get("p_layer2", 0.0) * w.get("layer2", 0.0)) +
            (r.get("p_layer3", 0.0) * w.get("layer3", 0.0))
        )
        return p_final, "|".join(present)

    out = probs.copy()
    blend = out.apply(_blend_row, axis=1, result_type="reduce")
    out["p_final"] = blend.apply(lambda x: x[0])
    out["used_layers"] = blend.apply(lambda x: x[1])
    out["risk_category"] = out["p_final"].apply(lambda x: _categorize(x, args.low_thr, args.high_thr))
    out["advice"] = out["risk_category"].apply(_advice_for)

    return out

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Final simple ensemble with auto fallback (features → probs).")

    # Mode A: single intake CSV scored by all three models
    ap.add_argument("--input_csv", default=None, help="Union-of-features intake CSV (features mode). If omitted, probs mode will be used.")
    ap.add_argument("--id_col", default="id", help="ID column name in --input_csv")
    ap.add_argument("--layer1_model", default="layer1_fox_out/model_layer1.pkl")
    ap.add_argument("--layer2_model", default="layer2_ppmi_out/model_layer2_ppmi.pkl")
    ap.add_argument("--layer3_model", default="layer3_shortened_out/model_layer3_shortened.pkl")

    # Mode B: per-layer probability CSVs
    ap.add_argument("--p1", default="layer1_fox_out/probs_layer1.csv")   # id, split, p_layer1
    ap.add_argument("--p2", default="layer2_ppmi_out/p_layer2_ppmi.csv") # PATNO, p_layer2
    ap.add_argument("--p3", default="layer3_shortened_out/p_layer3.csv") # id, p_layer3
    ap.add_argument("--id1", default="id")
    ap.add_argument("--id2", default="PATNO")
    ap.add_argument("--id3", default="id")
    ap.add_argument("--p1col", default="p_layer1")
    ap.add_argument("--p2col", default="p_layer2")
    ap.add_argument("--p3col", default="p_layer3")

    # Weights and bands
    ap.add_argument("--w1", type=float, default=DEFAULT_WEIGHTS["layer1"])
    ap.add_argument("--w2", type=float, default=DEFAULT_WEIGHTS["layer2"])
    ap.add_argument("--w3", type=float, default=DEFAULT_WEIGHTS["layer3"])
    ap.add_argument("--low_thr", type=float, default=DEFAULT_LOW_THR)
    ap.add_argument("--high_thr", type=float, default=DEFAULT_HIGH_THR)

    # Output
    ap.add_argument("--outdir", default="final_ensemble")

    return ap.parse_args()

# ---------- Main ----------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_weights = {"layer1": args.w1, "layer2": args.w2, "layer3": args.w3}
    with open(outdir / "weights.json", "w") as f:
        json.dump(base_weights, f, indent=2)
    print("[weights]", base_weights)

    combined = None

    # Try FEATURES MODE if input_csv provided
    if args.input_csv:
        try:
            combined = run_features_mode(args)
            if combined is not None:
                mode = "features"
        except Exception as e:
            print(f"[warn] Features mode failed: {e}")
            combined = None

    # If features not used / failed / all-NaN p_final, try PROBS MODE
    if combined is None:
        combined = run_probs_mode(args)
        mode = "probs" if combined is not None else None

    if combined is None:
        raise SystemExit("No usable output. Provide --input_csv (features mode) or valid --p1/--p2/--p3 files (probs mode).")

    out_csv = outdir / "combined_probs.csv"
    combined.to_csv(out_csv, index=False)

    with open(outdir / "config.json", "w") as f:
        json.dump({
            "mode": mode,
            "weights": base_weights,
            "low_thr": args.low_thr,
            "high_thr": args.high_thr,
            "files": {
                "input_csv": args.input_csv,
                "p1": args.p1, "p2": args.p2, "p3": args.p3
            }
        }, f, indent=2)

    print(f"✅ wrote {out_csv}")

if __name__ == "__main__":
    main()
