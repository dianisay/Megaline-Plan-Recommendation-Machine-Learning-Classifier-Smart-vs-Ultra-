# 📱 Smart vs Ultra: Megaline Plan Recommendation Model

## 📌 Introduction
Build a supervised classification model to **recommend the optimal Megaline plan (Smart vs Ultra)** from monthly usage behavior.  
Primary KPI: **Accuracy ≥ 0.75** on the held-out test set. Secondary checks include class-wise precision/recall and confusion matrix to ensure practical reliability.

---

## 🎯 Business goals
- **Plan assignment:** Predict the right plan to migrate users off legacy plans.
- **Operational impact:** Improve targeting for plan recommendations and reduce misclassification costs.
- **Model quality bar:** Meet or exceed **0.75 accuracy** on unseen data.

---

## 📂 Dataset
Each row represents one user’s monthly behavior:

- `calls` — number of calls  
- `minutes` — total call minutes  
- `messages` — number of SMS  
- `mb_used` — internet traffic in MB  
- `is_ultra` — target label (Ultra=1, Smart=0)

Source file: `datasets/users_behavior.csv`

---

## 🧭 Methodology

### 1) Data loading & checks
- **Schema & missing values:** Validate dtypes, nulls, and outliers.
- **Target distribution:** Inspect class balance (Smart vs Ultra) for proper evaluation.

### 2) Splits & baseline
- **Stratified split:** Train/Validation/Test (e.g., 60/20/20) with fixed `random_state`.
- **Baseline:** `DummyClassifier` (most_frequent and stratified) to set a floor for accuracy.

### 3) Modeling & tuning
- **Feature prep:** Keep raw features; **scale** only for distance/linear models.
- **Candidates:** Logistic Regression, k-NN, Decision Tree, Random Forest, Gradient Boosting.
- **Hyperparameters:** Grid/Random search on the validation set (no test leakage).  
  - Examples:  
    - Logistic: `C`, `penalty`  
    - k-NN: `n_neighbors`, `weights`  
    - Tree: `max_depth`, `min_samples_split`  
    - RandomForest: `n_estimators`, `max_depth`, `max_features`  
    - GradientBoosting: `n_estimators`, `learning_rate`, `max_depth`
- **Selection:** Pick the best validation accuracy; optionally refit on Train+Val before Test.

### 4) Evaluation
- **Primary:** Accuracy on the hold-out test set (target ≥ 0.75).
- **Diagnostics:** Confusion matrix; class precision/recall; ROC-AUC as a stability check.
- **Interpretability:** Feature importance (trees/boosting) or coefficients (logistic).

### 5) Sanity checks
- **Leakage guard:** Confirm no target leakage and scaler fit only on training folds.
- **Permutation test:** Shuffle labels to verify accuracy collapses to baseline.
- **Robustness:** Repeat with multiple seeds; report mean ± std of accuracy.

---

## 📊 Key analyses & visuals
- **EDA snapshots:** Distributions and correlations among `minutes`, `mb_used`, `calls`, `messages`.
- **Model comparison:** Validation accuracy by model/hyperparameters.
- **Final quality:** Test confusion matrix and per-class metrics; feature importance bar chart.

---

## ✅ Results summary
- **Best model:** <model_name> with tuned hyperparameters.  
- **Test accuracy:** <value> (≥ 0.75 target).  
- **Insights:** Which behaviors most strongly separate Ultra vs Smart (e.g., high `mb_used` or `minutes`).  
- **Next steps:** Threshold tuning by business cost, calibration if probabilities are used.

---

## 🧰 Tech stack
- **Python:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Utilities:** joblib (model persistence), mlxtend (optional for plots)

---

## 🚀 Repro steps
1. **Clone & env:** `pip install -r requirements.txt`
2. **Data:** Place `users_behavior.csv` in `datasets/`.
3. **Run:** Execute the notebook or `python src/train.py`
4. **Outputs:** Metrics in `reports/`, figures in `figures/`, model in `models/`

---

## 📈 Sample figures to include
- Validation accuracy per model  
- Test confusion matrix  
- Feature importance (tree-based) or coefficients (logistic)

---

## 🤝 Contact
Created by **Diana <Last Name>**  
🔗 [LinkedIn](https://linkedin.com/in/tuusuario) · 🌐 [Portfolio](https://tuportfolio.com)
