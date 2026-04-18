# AI-Based Autism Spectrum Disorder (ASD) Screening System

## Project Report

---

## 1. What is Autism Spectrum Disorder (ASD)?

Autism Spectrum Disorder (ASD) is a developmental condition that affects how a person communicates, interacts with others, and experiences the world around them. It is called a "spectrum" because it affects people in different ways and to different degrees — some individuals may need significant support in daily life, while others live independently with mild challenges.

Common signs of ASD include:

- Difficulty in social interactions (trouble making eye contact, understanding body language, or making friends)
- Repetitive behaviors (repeating words, following strict routines, or focused interest in specific topics)
- Sensitivity to sound, light, or textures
- Difficulty understanding other people's emotions or intentions

### Why is early screening important?

- ASD affects approximately **1 in 36 children** worldwide (CDC, 2023)
- Early diagnosis (before age 4) leads to significantly better outcomes through early intervention
- Traditional diagnosis relies on lengthy clinical observations by specialists, which can take **months to years** due to long waiting lists
- Many families, especially in rural or underserved areas, have limited access to specialists

**The core problem:** There is a massive gap between the number of children who need screening and the number of specialists available to screen them. This is where our AI system steps in.

---

## 2. How Can Machine Learning Help?

Machine Learning (ML) is a branch of Artificial Intelligence where computers learn patterns from data — instead of being explicitly programmed with rules, the system learns from examples.

Think of it like this: if you show a doctor thousands of patient records along with their diagnoses, the doctor develops an intuition for recognizing ASD traits. Machine learning does the same thing, but with mathematical precision and at scale.

### What our system does:

1. **Takes in patient data** — answers to a behavioral questionnaire, age, gender, family history, and clinical indicators
2. **Runs it through trained ML models** — these models have already learned patterns from 1,100 real cases
3. **Outputs a risk score** — a probability (0-100%) of ASD traits, categorized as Low, Moderate, or High Risk

This does **not** replace a doctor. It is a **screening tool** — like a thermometer that tells you if you might have a fever, not the diagnosis itself. It helps doctors and parents decide who should be referred for a full clinical evaluation.

---

## 3. Our Dataset — What Data Did We Use?

### Source

We used publicly available, anonymized datasets that have been used in ASD research. Our final dataset combines data from multiple sources into a single multimodal dataset.

### Dataset at a Glance

| Property | Value |
|---|---|
| Total samples | **1,100 individuals** |
| ASD positive (Class 1) | **393 (35.7%)** |
| No ASD (Class 0) | **707 (64.3%)** |
| Number of features | **35** |
| Train set (80%) | **880 samples** |
| Test set (20%) | **220 samples** |

### What features (columns) does the data contain?

We grouped our 35 features into meaningful categories:

#### A. AQ-10 Screening Questions (10 features)
The AQ-10 (Autism Quotient - 10 items) is a widely used, clinically validated quick screening questionnaire. Each question captures a specific behavioral trait:

| Feature | What it measures |
|---|---|
| A1_Score | Prefers doing things alone rather than with others |
| A2_Score | Prefers the same routine and gets upset by changes |
| A3_Score | Difficulty imagining what a fictional character thinks or feels |
| A4_Score | Gets easily absorbed in one thing at a time |
| A5_Score | Notices small sounds that others don't |
| A6_Score | Notices patterns (like numbers or textures) in things |
| A7_Score | Difficulty reading between the lines in conversation |
| A8_Score | Difficulty understanding unwritten social rules |
| A9_Score | Difficulty working out what someone is thinking or intending |
| A10_Score | Difficulty making new friends |

Each is scored as 0 (trait not present) or 1 (trait present).

#### B. Demographic Information (8 features)
| Feature | What it is |
|---|---|
| age | Age of the individual |
| gender | Male or Female |
| ethnicity | Ethnic background (encoded as numbers) |
| jundice | Whether the person was born with jaundice (a known risk factor) |
| austim | Whether a family member has ASD (genetic risk factor) |
| contry_of_res | Country of residence |
| age_desc | Age group category (child, adolescent, adult) |
| relation | Who completed the questionnaire (parent, self, etc.) |

#### C. Clinical and Other Indicators (17 features)
These come from clinical assessments and include neurological exam results, schooling history, and other medical indicators that provide additional context beyond the questionnaire.

### Data Quality Steps We Took

1. **Removed data leakage** — Some columns like "Autism_Diagnosis" and "Therapy_Progress" were essentially giving away the answer. We identified and removed 5 such leaking columns.
2. **Removed identifier columns** — Columns like patient ID, unnamed indices that carry no predictive value.
3. **Filtered to valid cases** — Our merged dataset had 2,100 rows, but 1,000 of those came from a completely different source with different column structures. We filtered down to the 1,100 consistent, binary-labeled cases.
4. **Handled missing values** — Used mean imputation (replacing missing values with the column average) for the small number of missing entries.
5. **80/20 Train-Test Split** — 880 samples for training, 220 for testing. The split is stratified, meaning both sets maintain the same ratio of ASD to non-ASD cases.

---

## 4. Our Machine Learning Models — Explained Simply

We trained and compared **4 different ML models**. Each has a different approach to learning patterns:

### Model 1: Random Forest

**Simple analogy:** Imagine asking 300 different doctors to independently look at a patient's data and vote on whether they show ASD traits. Each doctor sees a slightly different subset of the information. The final decision is based on majority vote.

**How it works:**
- Creates 300 "decision trees" — each tree is a series of yes/no questions about the data
- Each tree is trained on a random subset of the data and features
- Final prediction = majority vote of all 300 trees

**Why we used it:** Very robust, rarely overfits, handles mixed data types well, and tells us which features are most important.

**Our result:** 96.8% accuracy, 95.5% F1 score

---

### Model 2: XGBoost (Extreme Gradient Boosting)

**Simple analogy:** Instead of 300 independent doctors, imagine one doctor who makes an initial guess, then a second doctor who specifically focuses on correcting the first doctor's mistakes, then a third doctor who corrects the remaining errors, and so on for 300 rounds.

**How it works:**
- Builds trees sequentially, where each new tree learns from the errors of the previous ones
- Uses "gradient boosting" — a mathematical technique to minimize prediction errors step by step
- Very efficient and often wins ML competitions

**Why we used it:** Known for high accuracy on tabular (spreadsheet-like) data. Often the best-performing model in practice.

**Our result:** 97.3% accuracy, 96.2% F1 score

---

### Model 3: SVM (Support Vector Machine)

**Simple analogy:** Imagine plotting all patients on a graph. ASD patients cluster on one side, non-ASD on the other. SVM finds the best possible dividing line (or curve) between the two groups, with the widest possible margin.

**How it works:**
- Finds the optimal boundary (called a "hyperplane") that separates the two classes
- Uses a "kernel trick" (RBF kernel in our case) to handle cases where data isn't linearly separable — essentially projecting data into a higher dimension where a clean boundary exists
- Maximizes the margin between the closest points of each class

**Why we used it:** Excellent at finding complex, non-linear patterns. Works very well on medium-sized datasets.

**Our result:** 99.5% accuracy, 99.4% F1 score (our best model)

---

### Model 4: LightGBM (Light Gradient Boosting Machine)

**Simple analogy:** Similar to XGBoost (doctors correcting each other's mistakes), but using a faster, more memory-efficient approach. It's like the same team of doctors, but they've found shortcuts to reach conclusions faster.

**How it works:**
- Also uses gradient boosting like XGBoost
- Uses a "leaf-wise" growth strategy instead of "level-wise" — it grows the tree by splitting the leaf with the highest potential improvement first
- Optimized for speed and lower memory usage

**Why we used it:** Handles large datasets efficiently and often matches XGBoost's accuracy with faster training.

**Our result:** 95.0% accuracy, 93.1% F1 score

---

## 5. Results — How Well Do Our Models Perform?

### Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| **SVM** | **99.5%** | **100%** | **98.7%** | **99.4%** | **99.99%** |
| XGBoost | 97.3% | 96.2% | 96.2% | 96.2% | 99.8% |
| Random Forest | 96.8% | 97.4% | 93.7% | 95.5% | 99.8% |
| LightGBM | 95.0% | 92.5% | 93.7% | 93.1% | 99.3% |

### What Do These Metrics Mean?

- **Accuracy** — Out of all predictions, how many were correct? (SVM got 219 out of 220 right)

- **Precision** — When the model says "this person has ASD traits," how often is it correct? (SVM: 100% — it never falsely flagged a healthy person)

- **Recall** — Out of all actual ASD cases, how many did the model catch? (SVM: 98.7% — it missed only 1 out of 79 ASD cases)

- **F1 Score** — The balanced average of Precision and Recall. This is the single best metric when you care about both catching ASD cases AND not falsely alarming healthy individuals.

- **ROC AUC** — Measures the model's ability to distinguish between ASD and non-ASD across all possible thresholds. 1.0 is perfect; 0.5 is random guessing. All our models are above 0.99.

### Key Takeaway

SVM is our best-performing model with 99.5% accuracy. It correctly identified 78 out of 79 ASD cases in the test set while producing zero false positives — meaning every person it flagged as having ASD traits actually had ASD traits.

---

## 6. Explainability — Why Did the Model Decide This?

A common concern with AI in healthcare is the "black box" problem — the model gives an answer but we don't know why. We addressed this using **SHAP (SHapley Additive exPlanations)**.

### What is SHAP?

SHAP is a technique that explains each prediction by showing how much each feature contributed to the final decision. Think of it like a receipt — instead of just seeing the total bill, you see what each item cost.

For example, for a specific patient:
- "A7_Score = 1" pushed the prediction toward ASD by +15%
- "age = 25" pushed toward non-ASD by -3%
- "jundice = 1" pushed toward ASD by +8%

### Top Important Features (from our analysis)

Based on SHAP analysis of our Random Forest model, the most influential features in predicting ASD are:

1. **AQ-10 screening scores** (A1 through A10) — These behavioral questions are by far the strongest predictors, which aligns with clinical knowledge
2. **Age and demographics** — Age group and gender provide additional context
3. **Family history (austim)** — Having a family member with ASD is a known genetic risk factor
4. **Jaundice at birth (jundice)** — Neonatal jaundice has been studied as a potential risk factor

This is reassuring because our model's important features match what medical researchers already know about ASD risk factors.

---

## 7. System Architecture — How Everything Fits Together

```
                    +------------------+
                    |   Raw Datasets   |
                    | (AQ-10, Clinical,|
                    |  Eye, Motor)     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Preprocessing   |
                    | (Clean, Merge,   |
                    |  Remove Leakage) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Feature Engine  |
                    | (35 features)    |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v-----+  +-----v------+
     | Train (80%)|  | Test (20%) |  | Hyper-Param|
     | 880 samples|  | 220 samples|  |   Search   |
     +--------+---+  +------+-----+  +-----+------+
              |              |              |
     +--------v--------------v--------------v------+
     |              4 ML Models                     |
     | Random Forest | XGBoost | SVM | LightGBM    |
     +--------+------------------------------------+
              |
     +--------v---------+     +------------------+
     |   Evaluation     |     |   SHAP Analysis  |
     | (Metrics, ROC,   |<--->| (Explainability) |
     |  Confusion Matrix)|     +------------------+
     +--------+---------+
              |
     +--------v---------+
     |   Dashboard       |
     | (Streamlit Web UI)|
     | - Prediction Form |
     | - Model Analysis  |
     | - Data Explorer   |
     +-------------------+
```

---

## 8. The Dashboard — Our Web Application

We built an interactive web dashboard using **Streamlit** that allows users to:

### A. Make Predictions
- Fill in the AQ-10 questionnaire (10 yes/no behavioral questions)
- Enter demographics (age, gender, family history)
- Enter clinical scores
- Click "Predict" and instantly get:
  - ASD probability (0-100%)
  - Risk level (Low / Moderate / High)
  - Visual progress bar

### B. Analyze Model Performance
- Side-by-side comparison of all 4 models
- ROC curves showing how well each model separates ASD from non-ASD
- Precision-Recall curves
- SHAP plots showing which features matter most
- Confusion matrix showing exact prediction counts

### C. Explore the Dataset
- View data statistics and distributions
- Check correlations between features and the target
- Filter and download subsets of the data

---

## 9. Technical Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.14 | Core programming language |
| ML Models | scikit-learn, XGBoost, LightGBM | Model training and prediction |
| Deep Learning | PyTorch | LSTM/GRU models for temporal data |
| Explainability | SHAP | Understanding model decisions |
| Dashboard | Streamlit | Interactive web interface |
| API | FastAPI + Uvicorn | REST API for predictions |
| Experiment Tracking | MLflow | Logging experiments and parameters |
| Data Processing | Pandas, NumPy | Data manipulation and computation |
| Visualization | Matplotlib, Seaborn | Charts and plots |

---

## 10. Challenges We Faced and How We Solved Them

### Challenge 1: Data Leakage
**Problem:** Initial models showed 100% accuracy — which is unrealistic and suspicious. Investigation revealed that certain columns (like "Autism_Diagnosis" and "result") were directly derived from the target label, essentially giving the model the answer.

**Solution:** Identified and removed 5 leaking columns. After fixing this, models showed realistic performance (93-99.5%), which is still excellent.

### Challenge 2: Mixed Source Datasets
**Problem:** The merged dataset combined rows from different sources. One source (1,000 rows) had completely different columns filled compared to the other (1,100 rows), making them trivially separable by NaN patterns alone.

**Solution:** Filtered to the consistent binary-labeled subset (1,100 rows) and dropped columns with >80% missing values.

### Challenge 3: Class Imbalance
**Problem:** The dataset has 707 non-ASD vs 393 ASD cases (64:36 ratio), which can bias models toward predicting the majority class.

**Solution:** Used stratified train-test splitting to maintain the same class ratio in both sets, and evaluated using F1 score (which balances precision and recall) rather than just accuracy.

---

## 11. Limitations and Future Scope

### Current Limitations
- The dataset is relatively small (1,100 samples) — larger datasets would improve generalization
- The model is a screening tool, not a diagnostic tool — it should supplement, not replace, clinical judgment
- Currently uses only tabular/questionnaire data — eye-tracking and motor behavior models are built but not yet integrated into the main pipeline

### Future Enhancements
- Integrate LSTM model for motor behavior analysis (already built, needs pipeline integration)
- Integrate GRU model for eye-tracking signal analysis (already built)
- Build a multimodal fusion model that combines all data sources for more robust predictions
- Deploy the system as a cloud-hosted application for real-world pilot testing
- Expand the dataset with more diverse demographic representation

---

## 12. Conclusion

We built an end-to-end AI system for ASD risk screening that:

1. **Works accurately** — Our best model (SVM) achieves 99.5% accuracy with near-perfect precision and recall
2. **Is explainable** — SHAP analysis shows which features drive each prediction, building trust with clinicians
3. **Is accessible** — The Streamlit dashboard allows non-technical users to interact with the system
4. **Is methodologically sound** — We identified and fixed data leakage, used proper train-test evaluation, and compared multiple model architectures

The system demonstrates that machine learning can serve as a powerful, fast, and accessible first-line screening tool for ASD, potentially helping bridge the gap between the millions who need screening and the limited specialists available.

---

*This project is for research and educational purposes only. It does not constitute medical advice or diagnosis.*
