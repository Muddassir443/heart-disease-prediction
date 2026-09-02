# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease using the
UCI Heart Disease dataset, comparing multiple classification models and tuning
their hyperparameters for best performance.

## Dataset

The dataset (`heart-disease.csv`) contains 303 patient records with 13 clinical
features (age, sex, chest pain type, resting blood pressure, cholesterol, etc.)
and a binary target indicating the presence (1) or absence (0) of heart disease.

## Models Compared

- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Logistic Regression**

## Workflow

1. **Exploratory Data Analysis** — checked for missing values, visualized the
   relationship between resting blood pressure and cholesterol by disease status.
2. **Baseline comparison** — trained all three models on a 75/25 train-test split.
3. **Feature scaling** — applied `StandardScaler` to address KNN and Logistic
   Regression's sensitivity to feature magnitude, which improved KNN accuracy
   from ~70% to ~88% and resolved Logistic Regression's convergence warnings.
4. **Hyperparameter tuning** — used `RandomizedSearchCV` and `GridSearchCV`
   (5-fold cross-validation) to tune each model's hyperparameters.
5. **KNN neighbor analysis** — plotted train vs. test accuracy across
   `n_neighbors = 1–20` to visualize the bias-variance tradeoff and identify
   signs of overfitting/underfitting.

## Key Results

| Model | Accuracy (unscaled) | Accuracy (scaled) |
|---|---|---|
| KNN | 73.7% | 88.2% |
| Random Forest | 81.6% | 84.2% |
| Logistic Regression | 88.2%* | 88.2% |

*\*Unscaled Logistic Regression showed convergence warnings and required increased `max_iter`.*

## What This Project Demonstrates

- Importance of feature scaling for distance-based and gradient-based models
- Comparing multiple model families on the same dataset
- Cross-validated hyperparameter tuning
- Diagnosing and fixing common scikit-learn pitfalls (solver/penalty mismatches,
  convergence warnings, data leakage in scaling)

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (KNeighborsClassifier, RandomForestClassifier, LogisticRegression,
  GridSearchCV, RandomizedSearchCV, StandardScaler)
- Matplotlib

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook
```

Open the notebook and run all cells. Update the `pd.read_csv(...)` path to
point to your local copy of `heart-disease.csv`.
