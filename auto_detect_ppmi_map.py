python - <<'PY'
import argparse, os, re, json, pandas as pd

SRC="PPMI"         # your folder
OUT="overrides.json"

PREF = {
  "upsit": "University_of_Pennsylvania_Smell_ID_Test",
  "rbdsq": "PPMI_RBD_Sleep_Questionnaire",
  "scopa": "SCOPA-AUT",
  "moca":  "Montreal_Cognitive_Assessment",
  "demo":  "Demographics",
  "pdaq":  "PDAQ-27",
}

FEAT_PAT = {
  "hyposmia":               [r"upsit.*total", r"upsit.*score", r"smell.*total"],
  "rbd":                    [r"rbdsq.*total", r"rbd.*total"],
  "constipation":           [r"scopa.*gi.*constip", r"constip.*"],
  "urinary_dysfunction":    [r"scopa.*urin.*total", r"urin.*total"],
  "dizziness":              [r"scopa.*cv.*dizzi", r"dizzi.*(cv|cardio)"],
  "erectile_dysfunction":   [r"scopa.*sex.*male", r"erectile|ed"],
  "age":                    [r"age.*visit", r"age$"],
  "gender":                 [r"^sex$", r"gender$"],
  "memory_problems":        [r"pdaq.*mem", r"memory.*item"],
  "MOCA_TOTAL":             [r"moca.*total"],
  "PATNO":                  [r"^patno$"]
}

def find_col(cols, pats):
    cols=[c for c in cols if isinstance(c,str)]
    for p in pats:
        rx=re.compile(p, re.I)
        for c in cols:
            if rx.search(c): return c
    return None

def scan_one(path, keys):
    try: df=pd.read_csv(path, nrows=200, low_memory=False)
    except: return {}
    hits={}
    for k in keys:
        col=find_col(df.columns, FEAT_PAT[k])
        if col: hits[k]=col
    return hits

files=os.listdir(SRC)

def pick(prefix, keys):
    cands=[f for f in files if f.lower().startswith(prefix.lower()) and f.lower().endswith(".csv")]
    if not cands: return {}
    # prefer not Archived/Online
    cands.sort(key=lambda x: (("archived" in x.lower()) or ("online" in x.lower()), x.lower()))
    return scan_one(os.path.join(SRC,cands[0]), keys)

hits={}
for key, pref in PREF.items():
    keys=[]
    if key=="upsit": keys=["hyposmia"]
    if key=="rbdsq": keys=["rbd"]
    if key=="scopa": keys=["constipation","urinary_dysfunction","dizziness","erectile_dysfunction"]
    if key=="moca":  keys=["MOCA_TOTAL"]
    if key=="demo":  keys=["age","gender","PATNO"]
    if key=="pdaq":  keys=["memory_problems"]
    h=pick(pref, keys); hits.update(h)

overrides={}
for k in ["hyposmia","rbd","constipation","urinary_dysfunction","dizziness",
          "erectile_dysfunction","memory_problems","age","gender"]:
    if k in hits: overrides[k]=hits[k]

with open(OUT,"w") as f: json.dump(overrides,f,indent=2)
print("✅ wrote", OUT)
for k,v in overrides.items(): print(f"  - {k:22s} -> {v}")
print("\nDetected (reference for flags):")
print("  ID (PATNO) ->", hits.get("PATNO","<not found>"))
print("  MoCA total ->", hits.get("MOCA_TOTAL","<not found>"))
PY
