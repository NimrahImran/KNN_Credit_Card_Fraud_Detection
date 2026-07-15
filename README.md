# Credit Card Fraud Detection — KNN Classifier

An end-to-end machine learning pipeline that detects fraudulent credit card transactions using **K-Nearest Neighbors (KNN)**.

## 📌 Problem Statement
- **Objective:** Detect fraudulent credit card transactions based on anonymized transaction features
- **Dataset:** Credit Card Fraud dataset (`creditcard.csv`), balanced via undersampling to `creditcard_balanced.csv`
- **Problem Type:** Classification
- **Target Variable:** `Class` (0 = Legitimate, 1 = Fraud)
- **Business Relevance:** Early and accurate fraud detection helps financial institutions prevent monetary loss and protect customers from unauthorized transactions
- **Success Metric:** F1-score > 0.80 (due to class imbalance) → **Achieved: F1 = 0.9111**

## 📊 Dataset
- **Original dataset:** 67,434 transactions — highly imbalanced (67,264 legitimate vs. 169 fraud)
- **Balanced dataset:** Undersampled to 338 transactions (169 fraud, 169 legitimate) for fair model training
- **Features:** 28 anonymized PCA components (V1–V28) + `Amount` + `Time`
- **Dropped features:** `V17`, `V18` (highly correlated with other features, >0.95)
- No missing values or constant columns detected; 1 duplicate row removed

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

## 🔍 Pipeline / Methodology
1. **Data Loading & Class Balancing** — undersampled majority class to address severe imbalance
2. **Data Cleaning** — standardized column names, removed duplicate row, checked for missing/constant columns
3. **Exploratory Data Analysis** — dashboard overview, class distribution check, correlation analysis, skewness check
4. **Outlier Treatment** — boxplot-based capping applied to numeric features
5. **Feature Engineering** — removed highly correlated features (V17, V18)
6. **Preprocessing Pipeline** — median imputation + standard scaling (via `ColumnTransformer`)
7. **Train-Test Split** — 80/20 split (269 train / 68 test)
8. **Model Comparison** — benchmarked 10 classification algorithms using 5-fold cross-validation
9. **Model Training** — final pipeline trained using **KNN**
10. **Model Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report
11. **Sanity Checks** — train vs test score comparison to check for overfitting
12. **Cross Validation** — 5-fold CV for robustness check
13. **Hyperparameter Tuning** — GridSearchCV & RandomizedSearchCV to tune `n_neighbors`

## 📈 Results

| Metric | Value |
|--------|-------|
| **Accuracy (Test)** | 0.9118 |
| **Precision** | 0.9250 |
| **Recall** | 0.9118 |
| **F1-Score** | **0.9111** |
| Cross-Validation Mean Accuracy | 0.9376 (± 0.0345) |
| Best Hyperparameter (GridSearchCV) | `n_neighbors = 3` |
| Train Score vs Test Score | 0.9628 vs 0.9118 |

**Classification Report:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 (Legitimate) | 0.85 | 1.00 | 0.92 |
| 1 (Fraud) | 1.00 | 0.82 | 0.90 |

**Model Comparison (Top Results — CV Accuracy):**

| Model | CV Accuracy |
|-------|-------------|
| Random Forest | 0.9554 |
| Extra Trees | 0.9525 |
| AdaBoost | 0.9525 |
| Logistic Regression | 0.9524 |
| **KNN (selected)** | 0.9376 |

## 🔑 Key Insights
- **Best Model overall:** Random Forest achieved the highest cross-validation score (95.54%), slightly outperforming KNN
- KNN was selected as the final model and tuned using GridSearchCV, improving performance with `n_neighbors = 3

## 📂 Folder Structure
```
Credit-Card-Fraud-Detection/
│
├── data/
│   ├── creditcard.csv
│   └── creditcard_balanced.csv
├── knn-credit-card-fraud-detection.ipynb
└── README.md
```

## 👤 Author
**Nimra Muhammad Imran**
