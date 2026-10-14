# AIoT & DIC Multiple Linear Regression & Feature Selection Explorer

An interactive web application and machine learning workflow exploring Multiple Linear Regression (MultipleLR) and 5 Feature Selection strategies across **50 Startups** and **Boston Housing** datasets, plus a live client-side arbitrary dataset engine.

🌐 **Live Demo (GitHub Pages)**: [https://juic-12344.github.io/50-startup/](https://juic-12344.github.io/50-startup/)

---

## 🎯 Features & Capabilities

### 1. Dual Problem Exploration (+1 Point Bonus)
- **🏢 50 Startups Dataset**:
  - Predicts venture profit based on R&D Spend, Administration, Marketing Spend, and State dummy variables.
  - Demonstrates parsimony sweet spot at $k=2$ (R&D + Marketing Spend) with $R^2 \approx 94.74\%$.
- **🏡 Boston Housing Dataset**:
  - Predicts median home value (MEDV) across 13 socio-economic, structural, and environmental features.
  - Demonstrates textbook U-shaped overfitting curve: performance peaks at $k=9$ ($Adj\ R^2 = 68.98\%$) and degrades when noisy collinear features are added.

### 2. Five Feature Selection Strategies
1. **Lasso (L1 Regularization)**: Sparse coefficient shrinkage.
2. **Forward Stepwise Selection (SFS)**: Greedy forward search.
3. **Recursive Feature Elimination (RFE)**: Backward iterative elimination.
4. **ElasticNet (L1 + L2 Regularization)**: Balances sparsity and collinearity.
5. **Exhaustive Selection (Best Subset)**: Global combinatorial optimum for each $k$.

### 3. Interactive Analytics & Diagnostics
- **RMSE & R² Curves**: Tracks generalization error and highlights the parsimony sweet spot.
- **Feature Coefficients Bar Chart**: Magnitude and direction of active parameters.
- **Hypothesis Testing Table**: Standard errors, $t$-statistics, $p$-values ($***, **, *$), and 95% confidence intervals.
- **Live Predictor Playground**: Real-time sliders to forecast target values.
- **Residual & Outliers Diagnostic**: Tracks observations with standardized residuals $|z| > 2$.

### 4. In-Browser ML Engine for Arbitrary Datasets (+2 Points Bonus)
- Input any raw CSV URL (e.g. `https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`).
- Or drag-and-drop any custom local CSV file.
- Built-in pure JavaScript Ordinary Least Squares (OLS) matrix solver and forward feature selector running entirely client-side.

---

## 🚀 Local Quickstart

### Prerequisites
Python 3.8+ with standard data science packages:
```bash
pip install scikit-learn scipy statsmodels pandas numpy matplotlib
```

### Run Python ML Workflow & Server
```bash
# Run analysis and print comparative summary to terminal:
python main.py

# Launch the interactive web dashboard on localhost:8000:
python main.py --serve
```
Open [http://localhost:8000/index.html](http://localhost:8000/index.html) in your browser.
