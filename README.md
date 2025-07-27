# 🧠 NeuroScreen – AI for Early Parkinson’s Risk Detection  
⚡ Self-screen at home. 🔒 HIPAA-conscious. 📄 Auto PDF Reports.

**NeuroScreen** is a self-administered, AI-powered screening tool that predicts early Parkinson’s risk using clinically validated behavioral and motor survey inputs — no wearables, labs, or hospital visits required.

It combines **clinical-grade survey logic**, **real-time ML inference**, and **explainable AI** into one seamless Streamlit interface.

---

## ✅ Features

- 🎯 **Predicts Parkinson’s motor phenotype risk** (TD vs PIGD spectrum)
- 🧠 Converts 20+ neurobehavioral features into structured inputs
- 🧮 **Random Forest model**, trained on IRB-approved clinical data
- 📊 Integrated **SHAP explainability** to visualize feature influence
- 💾 **Zero data storage** — everything runs client-side
- 📄 **Auto-generates PDF reports** for participant or researcher use

---

## 🛠️ Tech Stack & Architecture

| Component       | Details |
|----------------|---------|
| UI/UX          | Streamlit, Multi-page Navigation, Sidebar Routing |
| Modeling       | `scikit-learn` Random Forest, `cloudpickle`, `SHAP` |
| Data Handling  | Preprocessing with custom Imputer + Encoder pipelines |
| Security       | HIPAA-conscious design (no server-side data capture) |
| Output         | Dynamic PDF generation using `fpdf` |
| Deployment     | Streamlit Cloud (Python 3.11) |

---

## 🧪 ML & AI Methodology

- Built on IRB-approved phenotype-labeled dataset with 30+ neuropsychological variables
- Binary classification using **motor phenotype ratio** (≥1.15 = TD, ≤0.90 = PIGD)
- Trained and cross-validated with `GridSearchCV` and stratified sampling
- Visualized SHAP values per session to **demystify predictions**
- Future roadmap includes:
  - Embedding clinical embeddings using `AutoML Tabular Transformers`
  - LLM-backed risk summarization using `OpenAI GPT API` (HIPAA-compatible mode)
  - Time-series adaptation for fNIRS & sensor integrations

---

## 🎯 Ideal Use Cases

- 🧪 Clinical trials (pre-screening participants by risk class)
- 🏥 Digital health product pilots & community screening
- 🧠 Neurology training (motor-cognitive phenotype awareness)
- 🧬 AI healthcare ethics demos (private, human-friendly ML)

---

## 💡 Why It Matters

> "This is what AI in healthcare should feel like — private, explainable, and centered on the human, not the hospital."

Unlike traditional tools built around institutional care, NeuroScreen empowers decentralized, early-stage self-screening — with ML that users can **see**, **understand**, and **trust**.

---

## 📁 Repo Structure

📦 neuroscreen-app/
┣ 📜 app.py # Main Streamlit entrypoint
┣ 📂 pages/ # Multi-step UI flow (Demographics, Motor, Cognitive, etc.)
┣ 📜 model_rf_final.pkl # Trained Random Forest classifier
┣ 📜 imputer.pkl # Pre-fitted imputer for missing data
┣ 📜 encoder.pkl # Categorical encoder
┣ 📜 requirements.txt # Runtime dependencies
┣ 📜 utils.py # Preprocessing + SHAP utilities
┗ 📄 README.md # This file


---

## 🚧 Coming Soon

- PDF + LLM-based auto summaries
- OAuth2 user session for secure study submissions
- Real-time analytics + confidence visualizations
- Clinical validation + external cohort replication (Phase 2)

---

## 🙋‍♀️ Built by

👩‍🔬 [Shruthi Reddy Vudem](https://www.linkedin.com/in/shruthireddyvudem/)  
Graduate Researcher – AI in Health, Saint Louis University  
Open to collaborations in AI4Health, Responsible AI, and ML deployment

