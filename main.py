# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 11:47:58 2025

@author: domin
"""

import os
from src.core import perform_prediction, risk_processing

# 1. Setup the Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def main():
    print("--- 1. Prediction ---")
    perform_prediction(
        model_name=os.path.join(DATA_DIR, "Smovement_I22_py.model"),
        new_dataset_name=os.path.join(DATA_DIR, "ver_A13.csv"),
        intervals=[12],
        pred_times=[12],
        saveresults=os.path.join(DATA_DIR, 'risk') 
    )

    print("\n--- 2. Risk Processing ---")
    # Capture the output in a variable named 'results'
    results = risk_processing(
        pred_name=os.path.join(DATA_DIR, "risk.pred.pkl"), 
        model_type='xgboost',
        centers=5,
        alpha=0.7,
        beta=0.3,
        original_csv = os.path.join(DATA_DIR, "A13_vertical_2016_2023_clipped.csv"),
        saveresults=os.path.join(DATA_DIR, "final_risk_v1")
    )

    # ---------------------------------------------------------
    # 4. RESULTS DISPLAY
    # ---------------------------------------------------------
    if results:
        print("\n--- Risk Table Head ---")
        # Check if key exists before printing to avoid errors
        if 'com12D' in results:
            print(results['com12D'][['PointID', 'RiskIndex', 'RiskCategory']].head())

        print("\n--- Max Risk per Point (First 5 rows) ---")
        if 'area_risk' in results:
            print(results['area_risk'][['PointID', 'MaxRiskIndex', 'MaxRiskCategory']].head())

        print("\n--- Clustering Performance Metrics ---")
        if 'metrics' in results:
            m = results['metrics']
            print(f"BSS / TSS Ratio: {m.get('bss_tss_ratio', 0):.2f}%")

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()