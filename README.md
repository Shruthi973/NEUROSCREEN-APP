# NeuroScreen – AI-Powered Parkinson’s Risk Screening (Multi-Dataset, Multi-Layer Pipeline)

**Live demo:** [https://neuroscreen-app-vsr-jas.streamlit.app/](https://neuroscreen-app-vsr-jas.streamlit.app/) 

⚡ Self-screen at home. 🔒 HIPAA-conscious. 📄 Auto PDF Reports.

<p align="center">
  <img src="./AI.jpg" width="600"/>
  <br>
  <em>🧠 NeuroScreen inspiration – Dopaminergic signaling and motor-cognitive interfaces in Parkinson’s Disease</em>
</p>
---

> NeuroScreen is a research-grade, explainable screening tool for early Parkinson’s risk estimation. It integrates evidence from multiple real-world datasets (Fox Insight, PPMI, and a REDCap clinical protocol) into a **three-layer** ML pipeline, surfaced through a single, user-friendly Streamlit app.

It combines **clinical-grade survey logic**, **real-time ML inference**, and **explainable AI** into one seamless Streamlit interface.
---

Presenting **NeuroScreen**, a multi-layer machine learning pipeline and web app that estimates early Parkinson’s disease (PD) risk using (i) large-scale questionnaire data, (ii) longitudinal clinical proxies, and (iii) a short REDCap-based clinical protocol.  

🔒 **Leak guards** enforce strict schema locks to prevent label leakage.  
⚡ **Fast inference** with pre-trained artifacts, deployable in Streamlit Cloud.  
📊 **Clinically interpretable** probabilities and risk tiers.  

---

## 🚀 Contributions

- 🔗 **Three-layer pipeline**:  
  - **Layer 1 (Fox Insight):** Self-report symptom proxies + demographics  
  - **Layer 2 (PPMI):** Clinical proxies for early cognitive impairment  
  - **Layer 3 (REDCap):** Shortened MDS-UPDRS + symptom items  

- 🛡️ **Schema locks & leak guards** for reproducibility  
- 📦 **Portable deployment** with Python 3.13 wheels  
- 👩‍⚕️ **Clinically legible UX** with plain-English prompts  

---

## 🧩 System Overview


Each layer outputs calibrated probabilities → merged into a **final screening estimate**.

---

## 📊 Data & Schemas

- **Fox Insight (L1):** smell, constipation, urinary, tremor, fatigue, memory, imbalance, age, sex  
- **PPMI (L2):** hyposmia proxy (`upsitorder`), RBD, urinary, vigilance, recall, age, sex  
- **REDCap (L3):** selected MDS-UPDRS I/II items, early symptoms (with `leak_guard` regex to exclude surrogates)  

⚖️ **Note:** Research-only tool, **not a diagnostic device**.  

---

## 🧪 Modeling Details

The pipeline follows a **layered evidence architecture**:  

- **Layer 1 (Fox Insight)**  
  *Input:* self-reported symptoms (hyposmia, constipation, urinary, tremor, imbalance, fatigue, memory, etc.) + demographics.  
  *Output:* calibrated probability of PD diagnosis (model artifact → `model_layer1.pkl`).  
  *Design note:* schema lock enforces exclusion of any variables directly correlated with diagnosis labels.  

- **Layer 2 (PPMI)**  
  *Input:* clinically validated proxies – hyposmia (`upsitorder`), REM sleep behavior (`RBD_SOURCE_OL`), urinary issues (`NP1URIN_OL`), vigilance (`MCAVIGIL`), recall (`DIFFRECALL`), plus age/sex.  
  *Output:* probability of early cognitive impairment (model artifact → `model_layer2_ppmi.pkl`).  
  *Design note:* acts as a **proxy detector** for prodromal PD cognitive features.  

- **Layer 3 (REDCap short protocol)**  
  *Input:* selected **MDS-UPDRS I/II** items & early symptom flags (subset of clinician-administered forms).  
  *Leak Guard:* regex exclusion for UPDRS totals, diagnosis surrogates.  
  *Output:* probability of PD screening-positive status (artifact → `model_layer3.pkl` with `layer3_feature_importance.csv`).  

- **Merger Layer**  
  *Inputs:* calibrated probabilities from L1, L2, L3.  
  *Strategy:* rule-based or weighted average (implemented in `merge_fox_v3_minimal.py`).  
  *Output:* a **final screening probability** + clinically legible risk tier (Low, Moderate, Elevated).  

---

## 🔮 Roadmap

This system is structured to **grow into a translational AI pipeline**. Upcoming phases:  

- **Explainability** → SHAP-based local attributions across all layers, plus global feature maps for clinical trust-building.  
- **Calibration** → post-hoc probability refinement using Platt scaling and isotonic regression per cohort.  
- **Dual-task integration** → extending REDCap protocol with cognitive-motor interference measures (walking while recalling, balance under distraction).  
- **External validation** → replication on unseen datasets & independent clinical cohorts.  
- **Mobile-first deployment** → lightweight on-device inference, edge-ready AI for at-home screening.  

---
## 📁 Repository Structure

📦 neuroscreen/  
┣ 📜 app.py # Streamlit entrypoint  
┣ 📜 requirements.txt # Dependencies (Python 3.13 wheels)  
┣ 📜 runtime.txt # python-3.13  
┣ 📜 schema_lock_layer1.json # Fox Insight schema lock  
┣ 📜 schema_lock_layer2.json # PPMI schema lock  
┣ 📜 schema_lock_layer3.json # REDCap schema lock  
┣ 📜 overrides.json # Variable alias mappings  
┣ 📜 model_layer1.pkl # Trained classifier (L1)  
┣ 📜 model_layer2_ppmi.pkl # Trained classifier (L2)  
┣ 📜 model_layer3.pkl # Trained classifier (L3)  
┣ 📜 metrics_summary.csv # Layer-wise metrics  
┣ 📜 merge_fox_v3_minimal.py # Probability merger utility  
┣ 📜 layer3_standalone.py # L3 training/inference utilities  
┗ 📜 auto_detect_ppmi_map.py # Heuristics for PPMI variable mapping  

---
## 📦 Requirements

This project is designed with strict environment pinning to ensure reproducibility and deployment stability.  
All dependencies are aligned with Python 3.13 wheels (no source builds).  

- **streamlit==1.36.0** → Interactive, multi-page app deployment  
- **numpy>=2.1,<2.3** → High-performance numerical computing  
- **pandas>=2.2.3,<2.4** → Data wrangling & schema validation  
- **scikit-learn>=1.6.1,<1.7** → ML pipeline (Random Forests, probability calibration, model persistence)  
- **scipy>=1.15.1,<1.16** → Statistical utilities & scientific transforms  
- **cloudpickle>=3.0.0** → Cross-version artifact serialization for ML models  
- **fpdf==1.7.2** → PDF export with participant- and clinician-facing summaries  

---

## 🙋 Maintainers

👩‍🔬 **Shruthi Reddy Vudem** – MS Health Data Science, Saint Louis University – Research Assistant  
📌 Expertise: Machine Learning, AI for Healthcare, Explainable ML, fNIRS-based neurocognitive modeling  

🤝 **Collaborators:** Jason Longhurst - Professor - Saint Louis University 

---

## 🌟 Why This Matters

NeuroScreen represents a **shift from single-cohort, opaque ML models** toward **multi-dataset, interpretable pipelines**.  
By harmonizing **patient self-report (Fox Insight)**, **clinical cohort proxies (PPMI)**, and **shortened clinic protocols (REDCap)**, this tool demonstrates how AI can:  

- Empower **early community-based screening**  
- Provide **clinician-facing interpretability**  
- Support **scalable digital health deployments**  

This is not just another ML model – it is a **research-grade AI framework** ready for adaptation in real-world translational neuroscience.  
