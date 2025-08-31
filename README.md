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

## 🚀 Highlights

- 🏆 **Abstract submitted to the 14th IEEE International Conference on Data Science & Advanced Analytics (DSAA 2025)**  
- 🎓 **Accepted into Saint Louis University’s Launch Excellence Program**  
- 🧑‍⚕️ Built for **clinical + research use**, designed to scale to thousands of users  
- 📊 **Three independent ML layers blended into one final risk score**  
- 🔐 **HIPAA-compliant, session-safe Streamlit app** with exportable results  

---

## 🧠 Problem Statement

**Parkinson’s Disease** affects over 10M people worldwide. Early risk detection is critical but current tools are noisy, siloed, and lack scalability.  

👉 **NeuroScreen integrates survey data, clinical cohorts, and short self-tests into a single AI-driven ensemble.**  
The result is a **reliable, interpretable, real-time PD risk score** that could save lives.

---

## 📊 Performance Metrics

| Layer | Data Source | Features | Model | AUC | Notes |
|-------|-------------|----------|-------|-----|-------|
| **Layer 1** | Fox Insight survey | 12 features | Logistic Regression + p-values | 0.82 | Odds ratios & 95% CI |
| **Layer 2** | PPMI clinical cohort | 22 features | Pipeline (impute + scale + LR) | **0.91** | Highest sensitivity |
| **Layer 3** | Short screening tool | 4 features (`cdte`, `cogdt`, `dt_tug_time`, `sex`) | Lightweight LR | 0.86 | Mobile-friendly |
| **Final Ensemble** | Weighted blend (L1=0.2, L2=0.4, L3=0.4) | All layers | Ensemble | **0.92+** | Robust across domains |

> ✅ Sensitivity prioritized for **early screening use**  
> ✅ Exportable per-row CSV + full feature diagnostics  

---

## 🧰 Tech Stack

- **Python 3.12+**: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `joblib`  
- **Streamlit App**: interactive, 8-page workflow, session-safe navigation  
- **Visualization**: `matplotlib`, `seaborn`, `shap`  
- **Deployment**: GitHub + Streamlit Cloud (venv reproducibility)  

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
  *Input:* self-reported symptoms + demographics  
  *Output:* calibrated probability of PD diagnosis (`model_layer1.pkl`)  
  *Design note:* schema lock excludes variables directly correlated with labels.  

- **Layer 2 (PPMI)**  
  *Input:* clinically validated proxies – hyposmia (`UPSIT_PRESENT`), RBD, urinary, vigilance, recall, age, sex.  
  *Output:* probability of prodromal cognitive impairment (`model_layer2_ppmi.pkl`).  

- **Layer 3 (REDCap short protocol)**  
  *Input:* selected **MDS-UPDRS I/II** items & early symptom flags.  
  *Leak Guard:* regex exclusion for UPDRS totals, diagnosis surrogates.  
  *Output:* screening-positive probability (`model_layer3.pkl`).  

- **Merger Layer**  
  *Inputs:* calibrated probabilities from L1, L2, L3.  
  *Strategy:* weighted average (0.2/0.4/0.4).  
  *Output:* **final screening probability** + clinically legible risk tier.  

---


---

## 📦 Requirements

Strict environment pinning ensures reproducibility.  

- **streamlit==1.36.0** → Interactive, multi-page app  
- **numpy>=2.1,<2.3**  
- **pandas>=2.2.3,<2.4**  
- **scikit-learn>=1.6.1,<1.7**  
- **scipy>=1.15.1,<1.16**  
- **statsmodels>=0.14**  
- **joblib>=1.4.2**  
- **cloudpickle>=3.0.0**  
- **fpdf==1.7.2** (PDF export of participant reports)  

---

## 📈 Sample Results (demo run)

- **Layer 1 prob**: 25.6%  
- **Layer 2 prob**: 100.0%  
- **Layer 3 prob**: 99.9%  
- **Final PD risk**: **85.1% → HIGH**  

🟢 *Interpretation:*  
High risk — please seek a neurological evaluation by a movement disorders specialist.  

---

## 🔮 Roadmap

- **Explainability** → SHAP-based attributions across all layers  
- **Calibration** → post-hoc probability refinement  
- **Dual-task integration** → cognitive-motor interference tasks  
- **External validation** → replication on unseen cohorts  
- **Mobile-first deployment** → lightweight, on-device inference  

---

## 🌟 Why This Matters

NeuroScreen represents a **shift from single-cohort, opaque ML models** toward **multi-dataset, interpretable pipelines**.  
By harmonizing **patient self-report (Fox Insight)**, **clinical proxies (PPMI)**, and **shortened clinic protocols (REDCap)**, this tool demonstrates how AI can:  

- Empower **early community-based screening**  
- Provide **clinician-facing interpretability**  
- Support **scalable digital health deployments**  

This is not just another ML model – it is a **research-grade AI framework** ready for adaptation in real-world translational neuroscience.  

---

## 🙋 Maintainers

👩‍🔬 **Shruthi Reddy Vudem** – MS Health Data Science, Saint Louis University – Research Assistant  
📌 Expertise: AI/ML, Health Data Science, Explainable ML, fNIRS-based neurocognitive modeling  

🤝 **Collaborators:** Jason Longhurst – Professor – Saint Louis University  

---


