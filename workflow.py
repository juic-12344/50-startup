"""
AIoT & DIC Assignment 2: Multiple Linear Regression & Feature Selection Workflow
Reusable ML pipeline supporting 50 Startups, Boston Housing, and any arbitrary dataset.
"""

import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


class RegressionPipeline:
    def __init__(self, df, target_col, categorical_cols=None, test_size=0.2, random_state=42):
        self.raw_df = df.copy()
        self.target_col = target_col
        self.categorical_cols = categorical_cols or []
        self.test_size = test_size
        self.random_state = random_state
        
        self.preprocess()
        
    def preprocess(self):
        df = self.raw_df.dropna().copy()
        
        # Target
        self.y = df[self.target_col].values
        
        # Feature dataframe
        X_raw = df.drop(columns=[self.target_col])
        
        # Categorical dummy encoding
        if self.categorical_cols:
            X_encoded = pd.get_dummies(X_raw, columns=self.categorical_cols, drop_first=True, dtype=float)
        else:
            # Check for object/category columns automatically
            obj_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
            if obj_cols:
                X_encoded = pd.get_dummies(X_raw, columns=obj_cols, drop_first=True, dtype=float)
            else:
                X_encoded = X_raw.astype(float)
                
        self.feature_names = list(X_encoded.columns)
        self.X = X_encoded.values
        
        # Train-Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Store feature stats for live predictor inputs (min, max, mean, step)
        self.feature_stats = {}
        for col in self.feature_names:
            vals = X_encoded[col].values
            vmin = float(vals.min())
            vmax = float(vals.max())
            vmean = float(vals.mean())
            is_dummy = set(vals).issubset({0.0, 1.0})
            step = 1.0 if is_dummy else round((vmax - vmin) / 100, 2)
            if step <= 0:
                step = 1.0
            self.feature_stats[col] = {
                "min": vmin,
                "max": vmax,
                "mean": round(vmean, 2),
                "step": step,
                "is_binary": is_dummy
            }

    def fit_and_evaluate(self, feature_indices):
        """Fit OLS on selected feature indices and return comprehensive metrics and coefficients."""
        p_sub = len(feature_indices)
        X_tr = self.X_train[:, feature_indices]
        X_te = self.X_test[:, feature_indices]
        n_train = len(self.y_train)
        
        # Fit Linear Regression
        lr = LinearRegression()
        lr.fit(X_tr, self.y_train)
        
        y_pred_te = lr.predict(X_te)
        y_pred_tr = lr.predict(X_tr)
        
        # Metrics on Test Set
        rmse = float(root_mean_squared_error(self.y_test, y_pred_te))
        mae = float(mean_absolute_error(self.y_test, y_pred_te))
        r2 = float(r2_score(self.y_test, y_pred_te))
        
        # Adjusted R2 on test set (or train set; sample used n_test, test_r2)
        n_test = len(self.y_test)
        if n_test - p_sub - 1 > 0:
            adj_r2 = float(1.0 - (1.0 - r2) * (n_test - 1.0) / (n_test - p_sub - 1.0))
        else:
            adj_r2 = r2
            
        # Statistical inference using Training set (standard statistical reporting)
        # Add intercept column
        X_design = np.column_stack([np.ones(n_train), X_tr])
        try:
            XtX_inv = np.linalg.pinv(X_design.T @ X_design)
            residuals_tr = self.y_train - y_pred_tr
            dof = max(1, n_train - p_sub - 1)
            sigma2 = np.sum(residuals_tr ** 2) / dof
            var_beta = np.diagonal(XtX_inv) * sigma2
            stderr = np.sqrt(np.maximum(var_beta, 1e-12))
            
            coefs_all = np.array([lr.intercept_] + list(lr.coef_))
            t_stats = coefs_all / stderr
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=dof))
            ci_low = coefs_all - stats.t.ppf(0.975, df=dof) * stderr
            ci_high = coefs_all + stats.t.ppf(0.975, df=dof) * stderr
            
            # F-statistic of the overall model
            ss_tot = np.sum((self.y_train - np.mean(self.y_train)) ** 2)
            ss_res = np.sum(residuals_tr ** 2)
            ss_reg = ss_tot - ss_res
            if p_sub > 0 and ss_res > 0:
                f_stat = float((ss_reg / p_sub) / (ss_res / dof))
                f_pval = float(1 - stats.f.cdf(f_stat, p_sub, dof))
            else:
                f_stat = 0.0
                f_pval = 1.0
        except Exception:
            stderr = np.zeros(p_sub + 1)
            t_stats = np.zeros(p_sub + 1)
            p_values = np.zeros(p_sub + 1)
            ci_low = np.zeros(p_sub + 1)
            ci_high = np.zeros(p_sub + 1)
            f_stat = 0.0
            f_pval = 1.0
            
        selected_names = [self.feature_names[i] for i in feature_indices]
        
        # Build coefficient list
        coefficients = []
        coefficients.append({
            "feature": "Intercept (β₀)",
            "coef": float(lr.intercept_),
            "stderr": float(stderr[0]),
            "tstat": float(t_stats[0]),
            "pvalue": float(p_values[0]),
            "ci_lower": float(ci_low[0]),
            "ci_upper": float(ci_high[0])
        })
        for i, fname in enumerate(selected_names):
            coefficients.append({
                "feature": fname,
                "coef": float(lr.coef_[i]),
                "stderr": float(stderr[i + 1]),
                "tstat": float(t_stats[i + 1]),
                "pvalue": float(p_values[i + 1]),
                "ci_lower": float(ci_low[i + 1]),
                "ci_upper": float(ci_high[i + 1])
            })
            
        # Residuals and Outlier detection on test set
        residuals_te = self.y_test - y_pred_te
        res_std = np.std(residuals_te) if np.std(residuals_te) > 0 else 1.0
        outliers = []
        for idx in range(len(self.y_test)):
            actual = float(self.y_test[idx])
            pred = float(y_pred_te[idx])
            res = float(residuals_te[idx])
            std_res = res / res_std
            outliers.append({
                "index": int(idx),
                "actual": round(actual, 2),
                "predicted": round(pred, 2),
                "residual": round(res, 2),
                "std_residual": round(std_res, 3),
                "is_outlier": bool(abs(std_res) > 2.0)
            })
            
        return {
            "k": p_sub,
            "features": selected_names,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pval,
            "coefficients": coefficients,
            "outliers": outliers
        }

    def run_all_methods(self):
        p = len(self.feature_names)
        results = {
            "Lasso": [],
            "Forward SFS": [],
            "RFE": [],
            "ElasticNet": [],
            "Exhaustive": []
        }
        
        # 1. Forward Stepwise Selection (SFS)
        current_features = []
        remaining = list(range(p))
        for k in range(1, p + 1):
            best_idx = None
            best_score = -float('inf')
            for candidate in remaining:
                cand_features = current_features + [candidate]
                lr = LinearRegression().fit(self.X_train[:, cand_features], self.y_train)
                score = lr.score(self.X_test[:, cand_features], self.y_test)
                if score > best_score:
                    best_score = score
                    best_idx = candidate
            current_features.append(best_idx)
            remaining.remove(best_idx)
            results["Forward SFS"].append(self.fit_and_evaluate(list(current_features)))
            
        # 2. RFE (Recursive Feature Elimination)
        # Fit full model, recursively drop feature with lowest absolute coef
        rfe_lr = LinearRegression()
        rfe_lr.fit(self.X_train, self.y_train)
        rfe = RFE(estimator=rfe_lr, n_features_to_select=1, step=1)
        rfe.fit(self.X_train, self.y_train)
        # Ranking: 1 is best, p is worst
        ranking_order = np.argsort(rfe.ranking_)  # indices in order of importance
        for k in range(1, p + 1):
            sel_idx = ranking_order[:k].tolist()
            results["RFE"].append(self.fit_and_evaluate(sel_idx))
            
        # 3. Lasso feature importance ranking
        # Standardize train features to find intrinsic importance path
        mean_tr = np.mean(self.X_train, axis=0)
        std_tr = np.std(self.X_train, axis=0)
        std_tr[std_tr == 0] = 1.0
        X_tr_std = (self.X_train - mean_tr) / std_tr
        
        lasso_cv = LassoCV(cv=5, random_state=self.random_state, max_iter=10000).fit(X_tr_std, self.y_train)
        lasso_importances = np.abs(lasso_cv.coef_)
        lasso_order = np.argsort(-lasso_importances).tolist()
        for k in range(1, p + 1):
            sel_idx = lasso_order[:k]
            results["Lasso"].append(self.fit_and_evaluate(sel_idx))
            
        # 4. ElasticNet feature ranking
        enet_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=self.random_state, max_iter=10000)
        enet_cv.fit(X_tr_std, self.y_train)
        enet_importances = np.abs(enet_cv.coef_)
        enet_order = np.argsort(-enet_importances).tolist()
        for k in range(1, p + 1):
            sel_idx = enet_order[:k]
            results["ElasticNet"].append(self.fit_and_evaluate(sel_idx))
            
        # 5. Exhaustive / Best Subset Selection
        # For each k, find combination with best test R^2 (or training MSE)
        for k in range(1, p + 1):
            combos = list(itertools.combinations(range(p), k))
            best_combo = None
            best_rmse = float('inf')
            # If p is large (e.g. > 15), limit combinations sample
            if len(combos) > 2000:
                # Top candidates from Forward + Lasso
                priority_set = set(current_features[:min(p, k + 4)])
                combos = [c for c in combos if any(idx in priority_set for idx in c)][:2000]
                
            for combo in combos:
                lr = LinearRegression().fit(self.X_train[:, combo], self.y_train)
                pred = lr.predict(self.X_test[:, combo])
                err = root_mean_squared_error(self.y_test, pred)
                if err < best_rmse:
                    best_rmse = err
                    best_combo = list(combo)
            results["Exhaustive"].append(self.fit_and_evaluate(best_combo))
            
        return results


def run_pipeline():
    print("Running Pipeline for 50 Startups...")
    df_startups = pd.read_csv('50_Startups.csv')
    pipe_startups = RegressionPipeline(
        df_startups, 
        target_col='Profit', 
        categorical_cols=['State'], 
        test_size=0.2, 
        random_state=0
    )
    res_startups = pipe_startups.run_all_methods()
    
    print("Running Pipeline for Boston Housing...")
    df_boston = pd.read_csv('BostonHousing.csv')
    pipe_boston = RegressionPipeline(
        df_boston, 
        target_col='medv', 
        test_size=0.2, 
        random_state=42
    )
    res_boston = pipe_boston.run_all_methods()
    
    # Bundle into comprehensive payload
    payload = {
        "datasets": {
            "startups": {
                "id": "startups",
                "name": "50 Startups",
                "title": "50 Startups: Profit Optimization & Multiple Linear Regression",
                "target": "Profit",
                "target_unit": "$ USD",
                "description": "Analyzes expenditure across R&D, Administration, and Marketing Spend with State locations (California, Florida, New York) to predict venture profit and find the most parsimonious predictive model.",
                "features": pipe_startups.feature_names,
                "feature_stats": pipe_startups.feature_stats,
                "models": res_startups
            },
            "boston": {
                "id": "boston",
                "name": "Boston Housing",
                "title": "Boston Housing: Urban Economics & MEDV Valuation Explorer",
                "target": "medv",
                "target_unit": "$1,000s",
                "description": "Investigates 13 socio-economic, environmental, and structural features (CRIM, RM, AGE, DIS, TAX, PTRATIO, LSTAT, etc.) across 506 Boston census tracts to predict Median House Value (MEDV) and diagnose multicollinearity.",
                "features": pipe_boston.feature_names,
                "feature_stats": pipe_boston.feature_stats,
                "models": res_boston
            }
        }
    }
    
    # Save as JSON and JS
    with open('model_data.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        
    js_content = "window.PROJECT_DATA = " + json.dumps(payload, indent=2) + ";\n"
    # Also provide backward-compatible MODEL_DATA pointing to 50 Startups
    js_content += "window.MODEL_DATA = window.PROJECT_DATA.datasets.startups.models;\n"
    
    with open('model_data.js', 'w', encoding='utf-8') as f:
        f.write(js_content)
        
    print("SUCCESS: model_data.json and model_data.js successfully generated!")
    print(f"50 Startups features: {len(pipe_startups.feature_names)}")
    print(f"Boston Housing features: {len(pipe_boston.feature_names)}")


if __name__ == '__main__':
    run_pipeline()
