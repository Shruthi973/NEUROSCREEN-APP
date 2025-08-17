# make_ppmi_overrides.py
# Scans PPMI/ for the correct columns and writes overrides.json (features only).
# Also prints the detected MoCA total column so you can pass it as --label/--moca_col.

import argparse, os, re, json, pandas as pd

PREF = {
    "upsit": "University_of_Pennsylvania_Smell_ID_Test",   # hyposmia
    "rbdsq": "PPMI_RBD_Sleep_Questionnaire",               # RBD total
    "scopa": "SCOPA-AUT",                                  # autonomic (urinary/constipation/dizziness/ED)
    "moca":  "Montreal_Cognitive_Assessment",              # MoCA total
    "demo":  "Demographics",                               # age/sex + PATNO
    "pdaq":  "PDAQ-27",                                    # memory complaints (optional)
}

PAT = {
    "hyposmia":             [r"upsit.*total", r"upsit.*score", r"smell.*total"],
    "rbd":                  [r"rbdsq.*total", r"rbd.*total"],
    "constipation":         [r"scopa.*gi.*constip", r"constip.*"],
    "urinary_dysfunction":  [r"scopa.*urin.*total", r"urin.*total"],
    "dizziness":            [r"scopa.*cv.*dizzi", r"dizzi.*(cv|cardio)"],
    "erectile_dysfunction": [r"scopa.*sex.*male", r"erectile|ed"],
    "memory_problems":      [r"pdaq.*mem", r"memory.*item"],
    "age":                  [r"age.*visit", r"^age(_at_visit)?$"],
    "gender":               [r"^sex$", r"^gender$"],
    # we also detect (but do not store in overrides) these two for you to use in flags:
    "MOCA_TOTAL":           [r"moca.*total"],
    "PATNO":                [r"^patno$"],
}

def first_match(cols, patterns):
    cols = [c for c in cols if isinstance(c, str)]
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in cols:
            if rx.search(c):
                return c
    return None

def scan(path, keys):
    try:
        df = pd.read_csv(path, nrows=200, low_memory=False)
    except Exception:
        return {}
    hits = {}
    for k in keys:
        col = first_match(df.columns, PAT[k])
        if col:
            hits[k] = col
    return hits

def pick(src, prefix, keys):
    files = [f for f in os.listdir(src) if f.lower().startswith(prefix.lower()) and f.lower().endswith(".csv")]
    if not files:
        return {}
    # prefer non-Archived / non-Online versions
    files.sort(key=lambda x: (("archived" in x.lower()) or ("online" in x.lower()), x.lower()))
    return scan(os.path.join(src, files[0]), keys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="PPMI", help="Folder containing PPMI CSVs")
    ap.add_argument("--out", default="overrides.json", help="Where to write the feature overrides")
    args = ap.parse_args()

    hits = {}
    hits.update(pick(args.src, PREF["upsit"], ["hyposmia"]))
    hits.update(pick(args.src, PREF["rbdsq"], ["rbd"]))
    hits.update(pick(args.src, PREF["scopa"], ["constipation","urinary_dysfunction","dizziness","erectile_dysfunction"]))
    hits.update(pick(args.src, PREF["pdaq"], ["memory_problems"]))
    hits.update(pick(args.src, PREF["demo"], ["age","gender","PATNO"]))
    hits.update(pick(args.src, PREF["moca"], ["MOCA_TOTAL"]))

    overrides = {k:v for k,v in hits.items() if k in {
        "hyposmia","rbd","constipation","urinary_dysfunction",
        "dizziness","erectile_dysfunction","memory_problems","age","gender"
    }}

    with open(args.out, "w") as f:
        json.dump(overrides, f, indent=2)

    print("✅ wrote", args.out)
    for k,v in overrides.items():
        print(f"  - {k:22s} -> {v}")

    print("\nDetected (use these in flags):")
    print("  ID (PATNO):     ", hits.get("PATNO","<not found>"))
    print("  MoCA total col: ", hits.get("MOCA_TOTAL","<not found>"))

if __name__ == "__main__":
    main()
