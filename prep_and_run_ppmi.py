# prep_and_run_ppmi.py
import os, sys, zipfile, subprocess

PPMI_DIR = "PPMI"  # where we'll look for csvs
ZIP_CANDIDATES = [
    os.path.expanduser("~/Downloads/PPMI.zip"),
    os.path.expanduser("~/Downloads/PPMI (1).zip"),
]

def have_csvs():
    return os.path.isdir(PPMI_DIR) and any(f.lower().endswith(".csv") for f in os.listdir(PPMI_DIR))

def ensure_ppmi_folder():
    os.makedirs(PPMI_DIR, exist_ok=True)
    if have_csvs():
        print(f"✅ Found CSVs in {PPMI_DIR}/")
        return
    # try to unzip a likely file
    for z in ZIP_CANDIDATES:
        if os.path.exists(z):
            print(f"📦 Unzipping {z} -> {PPMI_DIR}/")
            with zipfile.ZipFile(z, "r") as zip_ref:
                zip_ref.extractall(PPMI_DIR)
            break
    if not have_csvs():
        sys.exit(f"❌ No CSVs found in {PPMI_DIR}/. Put your PPMI .csv files there and rerun.")

def run_all_in_one():
    cmd = [
        sys.executable, "ppmi_layer2_all_in_one.py",
        "--src", PPMI_DIR,
        "--merge_out", "PPMI_clinical.csv",
        "--moca_col", "MOCA_TOTAL", "--moca_cut", "26",
        "--idcol", "PATNO", "--label_name", "EarlyCogImp",
        "--map", "ppmi_map.json",
        "--model_out", "model_layer2_ppmi.pkl",
        "--probs_out", "p_layer2_ppmi.csv",
    ]
    print("▶", " ".join(cmd))
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        sys.exit("❌ Pipeline failed.")
    print("🎉 Done.")

if __name__ == "__main__":
    ensure_ppmi_folder()
    run_all_in_one()
