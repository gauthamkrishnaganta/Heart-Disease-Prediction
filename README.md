# 🫀 Heart Disease Classification with Random Forest

An end-to-end Machine Learning classification pipeline designed to predict the presence of heart disease in patients based on various clinical and physiological features.

---

## 📌 Project Overview
- **Dataset:** Heart Disease Dataset (`heart.csv`) with 303 patient records and 13 clinical features.
- **Model:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`).
- **Optimization:** Hyperparameter tuning using `RandomizedSearchCV` (9-Fold Cross Validation).
- **Baseline Accuracy:** ~81.97%
- **Optimized Test Accuracy:** **85.25%**
- **Model Export:** Serialized using `pickle` as `heart.pkl` for downstream deployment.

---

## 📊 Dataset Features

The dataset contains the following attributes used to predict the `target` variable (0 = No Heart Disease, 1 = Heart Disease):

| Feature | Description |
|---|---|
| `age` | Age of the patient (in years) |
| `sex` | Sex (1 = Male, 0 = Female) |
| `cp` | Chest pain type (values 0–3) |
| `trestbps` | Resting blood pressure (in mm Hg on admission) |
| `chol` | Serum cholesterol level in mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting electrocardiographic results (values 0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | The slope of the peak exercise ST segment (values 0–2) |
| `ca` | Number of major vessels (0–4) colored by flourosopy |
| `thal` | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect) |
| **`target`** | **Class label (1: Disease present, 0: Disease absent)** |

---

## 🛠️ Project Workflow

1. **Data Ingestion & Cleaning:**
   - Loaded and inspected `heart.csv` using Pandas.
   - Verified that there are no missing (`NaN`) values.
2. **Exploratory Data Analysis (EDA):**
   - Correlation Matrix Heatmap generated using Seaborn to identify relationships between features and target.
   - Target distribution visualization.
3. **Data Splitting:**
   - Split features (`X`) and label (`y`) using an 80/20 train-test ratio (`random_state=42`).
4. **Baseline Model:**
   - Initialized `RandomForestClassifier(n_estimators=20)`.
   - Baseline evaluation: **81.97%** accuracy.
5. **Hyperparameter Tuning:**
   - Used `RandomizedSearchCV` across parameters such as `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`, `criterion`, and `bootstrap`.
   - Resulted in an enhanced test accuracy of **85.25%**.
6. **Serialization:**
   - Saved the trained model as `heart.pkl` for quick loading in web APIs or dashboards (e.g., Streamlit / Flask).

---

## 📈 Evaluation Results

### Confusion Matrix (Tuned Model)
