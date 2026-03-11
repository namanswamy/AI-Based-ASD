# 🧠 AI-Based Autism Spectrum Disorder (ASD) Research Platform

An end-to-end **AI-powered Autism Spectrum Disorder (ASD) screening system** that combines behavioral questionnaires, clinical indicators, and multimodal machine learning models to estimate ASD risk.

This project demonstrates a **complete AI research pipeline**, including:

* Data preprocessing
* Feature engineering
* Machine learning model training
* Model evaluation
* Explainable AI (SHAP)
* FastAPI inference service
* Streamlit research dashboard

The system provides **ASD risk prediction and interpretability tools** suitable for research and experimentation.

---

# 🚀 Features

### 🧠 Machine Learning Models

Multiple models are implemented for ASD prediction:

* Random Forest
* XGBoost
* Support Vector Machine (SVM)
* LightGBM

Additional experimental models:

* LSTM (Motor behavior patterns)
* GRU (Eye movement patterns)

---

### 📊 Explainable AI

The project integrates **SHAP explainability** to understand model predictions.

Capabilities:

* Global feature importance
* SHAP summary plots
* Model interpretability for research

---

### 🌐 API Server

A **FastAPI server** exposes prediction endpoints.

Capabilities:

* Real-time ASD risk prediction
* REST API for integration
* Swagger interactive documentation

---

### 🖥 Research Dashboard

A **Streamlit dashboard** allows researchers to interact with the system.

Dashboard modules:

* ASD risk prediction form
* Model performance analysis
* Dataset exploration

---

# 🏗 System Architecture

```
User
  │
  ▼
Streamlit Dashboard
  │
  ▼
FastAPI Inference Server
  │
  ▼
Machine Learning Models
(RandomForest / XGBoost / SVM / LightGBM)
  │
  ▼
Prediction + Explainability
```

---

# 📁 Project Structure

```
AI-Based-ASD/

api/
│
├── server.py
├── inference.py
└── schemas.py

dashboard/
│
├── app.py
└── pages/
    ├── prediction.py
    ├── model_analysis.py
    └── dataset_explorer.py

data/
│
├── raw/
└── processed/

preprocessing/
│
├── tabular_preprocessor.py
├── clinical_preprocessor.py
├── eye_preprocessor.py
├── motor_preprocessor.py
└── merge_datasets.py

models/
│
├── classical_models.py
├── lstm_models.py
├── gru_models.py
└── saved/

training/
│
├── train_tabular.py
├── train_motor.py
├── train_eye.py
└── hyperparameter_search.py

evaluation/
│
├── evaluate_models.py
├── model_comparison.py
├── roc_analysis.py
├── pr_curve.py
└── confusion_matrix_plot.py

explainability/
│
├── shap_analysis.py
└── model_explainer.py

utils/
│
├── config.py
├── metrics.py
├── logger.py
└── model_utils.py

outputs/
│
├── plots/
└── reports/

requirements.txt
README.md
```

---

# 🧪 Machine Learning Pipeline

### 1️⃣ Data Preprocessing

Raw datasets are cleaned and transformed.

Scripts:

```
preprocessing/tabular_preprocessor.py
preprocessing/clinical_preprocessor.py
preprocessing/eye_preprocessor.py
preprocessing/motor_preprocessor.py
```

---

### 2️⃣ Feature Engineering

Generated features include:

* Questionnaire scores
* Behavioral indicators
* Clinical measures
* Statistical movement features

---

### 3️⃣ Model Training

Training scripts:

```
training/train_tabular.py
training/train_motor.py
training/train_eye.py
```

Models trained:

* Random Forest
* XGBoost
* SVM
* LightGBM

---

### 4️⃣ Model Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score

Scripts:

```
evaluation/evaluate_models.py
evaluation/model_comparison.py
```

---

### 5️⃣ Explainable AI

Explain predictions using SHAP.

Scripts:

```
explainability/model_explainer.py
```

Generated outputs:

```
outputs/plots/
    shap_summary.png
    shap_bar.png
    model_feature_importance.png
```

---

# ⚙ Installation

### 1️⃣ Clone the Repository

```
git clone https://github.com/yourusername/asd-ai-platform.git
cd asd-ai-platform
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate:

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

# ▶ Running the Project

## 1️⃣ Run Preprocessing

```
python -m preprocessing.tabular_preprocessor
python -m preprocessing.clinical_preprocessor
python -m preprocessing.eye_preprocessor
python -m preprocessing.motor_preprocessor
python -m preprocessing.merge_datasets
```

---

## 2️⃣ Train Models

```
python -m training.train_tabular
```

Optional:

```
python -m training.hyperparameter_search
```

---

## 3️⃣ Evaluate Models

```
python -m evaluation.evaluate_models
```

---

## 4️⃣ Run Explainability

```
python -m explainability.model_explainer
```

---

# 🌐 Run the API Server

```
uvicorn api.server:app --reload
```

Open API documentation:

```
http://127.0.0.1:8000/docs
```

---

# 🖥 Run the Dashboard

```
streamlit run dashboard/app.py
```

Open:

```
http://localhost:8501
```

---

# 📊 Example Prediction

Input features are generated from questionnaire responses and behavioral indicators.

Example feature vector:

```
[0.23, 0.51, 0.67, ..., 0.43]
```

Output:

```
ASD Probability: 0.72
Risk Level: High
```

---

# 📈 Example Outputs

The system generates:

```
outputs/

plots/
  roc_curve.png
  confusion_matrix.png
  shap_summary.png

reports/
  model_comparison.csv
```

---

# 🔬 Research Applications

This platform can be used for:

* ASD behavioral screening research
* Machine learning experimentation
* Explainable AI studies
* Healthcare decision-support prototyping

---

# ⚠ Disclaimer

This system is **not a medical diagnostic tool**.
It is intended for **research and educational purposes only**.

---

# 👨‍💻 Author

Developed as an **AI research platform for Autism Spectrum Disorder prediction** using machine learning and explainable AI techniques.