# gen_schema_lock_layer1.py
import pandas as pd, json

DATA="FoxInsight.csv"
IDCOL="fox_insight_id"
LABEL="CurrPDDiag"
REQUIRED_FEATURES=[
  "ImpactMoveHyposmia","ImpactMoveConstipation","Anxiety","ImpactMoveUrinary",
  "ImpactMoveTremor","ImpactThinkBodyFatigue","ImpactThinkDizzy",
  "ImpactMoveImbalance","ImpactThinkMemory","MoveSaliva","age","Sex"
]

df=pd.read_csv(DATA, nrows=2000, low_memory=False)  # sample is enough for dtypes
missing=[c for c in [IDCOL,LABEL]+REQUIRED_FEATURES if c not in df.columns]
if missing: raise SystemExit(f"❌ Missing in {DATA}: {missing}")

schema={
  "data": DATA,
  "idcol": IDCOL,
  "label": LABEL,
  "features": REQUIRED_FEATURES,
  "dtypes": {c:str(df[c].dtype) for c in [IDCOL,LABEL]+REQUIRED_FEATURES}
}
with open("schema_lock_layer1.json","w") as f: json.dump(schema,f, indent=2)
print("✅ wrote schema_lock_layer1.json")
