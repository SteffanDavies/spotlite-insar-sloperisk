# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 11:47:58 2025

@author: domin
"""

import os
import pandas as pd
from src.core import perform_prediction, risk_processing

# ---------------------------------------------------------
# 1. SETUP PARAMETERS
# ---------------------------------------------------------
# Define paths relative to the project root or absolute paths
BASE_DIR = os.getcwd() 
DATA_DIR = os.path.join(BASE_DIR, "data") # Assumes you put CSVs in a 'data' folder
MODEL_FILE = "Smovement_I22_py.model"
INPUT_CSV = "ver_A13.csv" 

# ---------------------------------------------------------
# 2. RUN PREDICTION
# ---------------------------------------------------------
print("--- Starting Prediction ---")
predict_movement = perform_prediction(
        model_name=os.path.join(DATA_DIR, "Smovement_I22_py.model"),
        new_dataset_name=os.path.join(DATA_DIR, "ver_A13.csv"),
        intervals=[12],
        pred_times=[12],
        saveresults=os.path.join(DATA_DIR, 'risk') # Output: data/risk.pred.pkl
    )

# ---------------------------------------------------------
# 3. RUN RISK PROCESSING
# ---------------------------------------------------------
print("--- Starting Risk Processing ---")
results = risk_processing(
    pred_name="risk.pred.pkl", # Using the result from step 2
    model_type='xgboost', # Ensure this matches the algorithms
    centers=5,
    nstart=25,
    alpha=0.7,        # 70% weight on current movement magnitude
    beta=0.3,         # 30% weight on historical behavior (slope class)
    original_csv=os.path.join(DATA_DIR, "A13_vertical_2016_2023_clipped.csv"),
    saveresults= os.path.join(DATA_DIR, "final_risk_v1") # SAVES: final_risk_v1.pkl, final_risk_v1_spatial.gpkg, plots...
)

# ---------------------------------------------------------
# 4. RESULTS
# ---------------------------------------------------------
print("\n--- Risk Table Head ---")
print(results['com12D'][['PointID', 'RiskIndex', 'RiskCategory']].head())


# B. Check the Aggregated Area Risk
print("\n--- Max Risk per Point (First 5 rows) ---")
print(results['area_risk'][['PointID', 'MaxRiskIndex', 'MaxRiskCategory']].head())

# C. Check Cluster Centers (Returned as DataFrame)
print("\n--- PCA Cluster Centers ---")
print(results['cluster_centers'])

# D. Check Clustering Performance Metrics
print("\n--- Clustering Performance Metrics ---")
metrics = results['metrics']
print(f"Total Sum of Squares (TSS): {metrics['tss']:.2f}")
print(f"Within-Cluster SS  (WSS): {metrics['wss']:.2f}")
print(f"Between-Cluster SS (BSS): {metrics['bss']:.2f}")
print(f"BSS / TSS Ratio:          {metrics['bss_tss_ratio']:.2f}%")

# E. Check Cluster Sizes and Proportions
print("\n--- Cluster Sizes ---")
print(results['cluster_counts'])

print("\nProcessing Complete.")