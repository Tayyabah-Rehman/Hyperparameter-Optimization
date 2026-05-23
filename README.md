# 🏠 Linear-Regression

> Linear Regression Development & Evaluation Pipeline | Implements scikit-learn LinearRegression with comprehensive regression metrics calculation | Features predicted vs actual scatter plot with y=x reference line for visual model assessment.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ECC71?style=for-the-badge)

---

## 📌 About

This project implements a complete **Linear Regression pipeline** on the California Housing dataset to predict median house values. It covers data loading, train/test splitting, model training, evaluation using three standard metrics, and a scatter plot visualization of predicted vs. actual values.

**Course:** Machine Learning — Task 3
**Author:** Tayyabah Rehman
**Date:** May 2026

---

## 📊 Results

| Metric | Score | Meaning |
|--------|-------|---------|
| RMSE | 0.7456 | Average error ~$74,560 per prediction |
| MAE | 0.5332 | Typical prediction error ~$53,320 |
| R² | 0.5758 | Model explains 57.6% of variance |

---

## 📉 Predicted vs Actual Plot

![Predicted vs Actual](predicted_vs_actual.png)

> Red dashed line = perfect prediction (y = x). Points closer to the line = more accurate predictions.

---

## 📁 Project Structure

```
Linear-Regression/
│
├── Task3_ML_Tayyabah_Rehman.ipynb   # Main notebook
├── predicted_vs_actual.png          # Output scatter plot
└── README.md                        # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Linear-Regression.git
cd Linear-Regression
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

---

## ▶️ How to Run

### Jupyter Notebook
```bash
jupyter notebook Task3_ML_Tayyabah_Rehman.ipynb
```
Then: **Kernel → Restart & Run All**

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lPowg0NVOucV_45TtO85QCs6I02PZ9uE)

> No data files needed — dataset loads automatically from Scikit-learn.

---

## 🔢 Dataset

| Property | Value |
|----------|-------|
| Source | `sklearn.datasets.fetch_california_housing` |
| Rows | 20,640 |
| Features | 8 numerical |
| Target | `MedHouseVal` (median house value in $100,000s) |
| Missing values | 0 |
| Train / Test | 16,512 / 4,128 (80/20, random_state=42) |

---

## 🔧 Pipeline Steps

```python
# 1. Load dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# 2. Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

# 5. Plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.savefig('predicted_vs_actual.png', dpi=300, bbox_inches='tight')
```

---

## 🧰 Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and handling |
| `numpy` | RMSE calculation |
| `scikit-learn` | Model, splitting, metrics |
| `matplotlib` | Scatter plot visualization |

---

⭐ Star this repo if you found it useful!
