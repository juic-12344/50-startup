"""
=============================================================================
AIoT & DIC Assignment 2: Multiple Linear Regression & Feature Selection
Dual Dataset Investigation: 50 Startups vs. Boston Housing
=============================================================================
This script:
1. Executes the reusable ML pipeline for both 50 Startups and Boston Housing.
2. Compares feature selection methods: Lasso, Forward SFS, RFE, ElasticNet, Exhaustive.
3. Provides an in-depth analytical report recommending which dataset to upload.
4. Generates data for the interactive web dashboard (index.html).
5. Provides an optional one-click local server launcher to view the dashboard.
=============================================================================
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import pandas as pd
from workflow import run_pipeline, RegressionPipeline

# Ensure safe UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def print_banner(text):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def analyze_dataset_summary(pipeline, dataset_name, results):
    print_banner(f"Dataset Analysis: {dataset_name}")
    print(f"Total Samples : {len(pipeline.y)}")
    print(f"Total Features: {len(pipeline.feature_names)}")
    print(f"Target Column : {pipeline.target_col}")
    print(f"Features List : {pipeline.feature_names}")
    
    print("\n--- Feature Selection Trajectory (Forward Stepwise Selection) ---")
    headers = ["k", "Selected Feature", "Test RMSE", "Raw R^2 (%)", "Adj R^2 (%)", "F-Stat", "p-Value"]
    rows = []
    
    steps = results["Forward SFS"]
    for s in steps:
        last_feature = s["features"][-1]
        p_val_str = "< 0.0001" if s["f_pvalue"] < 0.0001 else f"{s['f_pvalue']:.4f}"
        rows.append([
            f"k={s['k']}",
            last_feature[:25],
            f"{s['rmse']:.2f}",
            f"{s['r2'] * 100:.2f}%",
            f"{s['adj_r2'] * 100:.2f}%",
            f"{s['f_stat']:.1f}",
            p_val_str
        ])
        
    df_report = pd.DataFrame(rows, columns=headers)
    print(df_report.to_string(index=False))
    
    # Identify Sweet Spot (Max Adjusted R2)
    best_step = max(steps, key=lambda x: x["adj_r2"])
    print(f"\n>> Parsimony Sweet Spot: k = {best_step['k']} features")
    print(f"   Features: {best_step['features']}")
    print(f"   Test RMSE: {best_step['rmse']:.2f} | Adj R^2: {best_step['adj_r2'] * 100:.2f}%")


def print_topic_recommendation():
    print_banner("Recommendation: Which Topic to Upload to Kaggle / Coursework?")
    print("""
[Decision Criteria & Comparative Evaluation]

1. 50 Startups:
   - Dimension: 50 observations, 5 features (after dummy encoding).
   - Character: High initial R^2 (~94.6% with R&D Spend alone). Adding Marketing Spend
     achieves the sweet spot (k=2, RMSE=8,198.8, R^2=94.74%).
   - Limitation: Small dataset (50 rows) and only 10 test samples in an 80/20 split.
     Administration and State variables have high p-values (insignificant).

2. Boston Housing (RECOMMENDED FOR FINAL UPLOAD):
   - Dimension: 506 observations, 13 features.
   - Character: Strong statistical depth, substantial sample size (102 test rows).
     Features span socio-economic (LSTAT, PTRATIO), structural (RM, AGE, TAX),
     and environmental (NOX, CRIM, DIS) domains.
   - Parsimony Demonstration: Exhibits a textbook U-shaped error curve:
     - k=1 (LSTAT alone): RMSE = 5.79, Adj R^2 = 53.8%
     - k=8/9 (Sweet Spot): RMSE = 4.54, Adj R^2 = 69.5%
     - k=13 (All features): RMSE increases back to 4.93, Adj R^2 falls to 61.98%!
     This clearly proves the danger of multicollinearity and overfitting in ML.

>> FINAL VERDICT:
   Upload **Boston Housing** for your final submission or report! It provides far more
   rigorous evidence of feature selection algorithms, parsimony trade-offs, and
   multicollinearity diagnostics.
""")


def start_server_and_browser(port=8000):
    print_banner(f"Launching Interactive Dashboard at http://localhost:{port}")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    handler = http.server.SimpleHTTPRequestHandler
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Server started at: http://localhost:{port}/index.html")
            print("Press Ctrl+C to stop the server.")
            webbrowser.open(f"http://localhost:{port}/index.html")
            httpd.serve_forever()
    except OSError:
        # Port already in use, try opening browser directly
        print(f"Port {port} in use or already running. Opening browser directly...")
        webbrowser.open(f"http://localhost:{port}/index.html")


def main():
    print_banner("AIoT & DIC Assignment 2: Multiple Linear Regression Workflow")
    
    # 1. Run Pipeline to generate model_data.json and model_data.js
    run_pipeline()
    
    # 2. Load and inspect
    df_startups = pd.read_csv('50_Startups.csv')
    pipe_startups = RegressionPipeline(df_startups, 'Profit', ['State'], test_size=0.2, random_state=0)
    res_startups = pipe_startups.run_all_methods()
    analyze_dataset_summary(pipe_startups, "50 Startups (Venture Profit)", res_startups)
    
    df_boston = pd.read_csv('BostonHousing.csv')
    pipe_boston = RegressionPipeline(df_boston, 'medv', test_size=0.2, random_state=42)
    res_boston = pipe_boston.run_all_methods()
    analyze_dataset_summary(pipe_boston, "Boston Housing (MEDV Valuation)", res_boston)
    
    # 3. Print Upload Recommendation
    print_topic_recommendation()
    
    print("\n[+] Web Application Files Created:")
    print("    - index.html   : Main interactive dashboard")
    print("    - style.css    : Premium dark glassmorphism styling")
    print("    - app.js       : In-browser ML engine, Chart.js managers & dual-problem controller")
    print("    - model_data.js: Precomputed regression results for both problems")
    print("    - workflow.py  : Reusable Python pipeline")
    
    # Check if user passed --serve
    if "--serve" in sys.argv:
        start_server_and_browser()


if __name__ == '__main__':
    main()
