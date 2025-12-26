# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 11:47:24 2025

@author: domin
"""

# --- FILE: insar_functions.py ---

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time
from datetime import timedelta
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold, cross_validate, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import geopandas as gpd
import string

# IMPORT FROM YOUR UTILS FOLDER
from src.utils.helpers import (
    load_data,
    extract_data,
    sliding_window, 
    process_data, 
    filter_by_time_intervals, 
    rename_and_clean_columns,
    calculate_statistics,
    com_datasets
)


# ==========================================
# 1. DEFAULT CONFIGURATION
# ==========================================
# This acts as your default settings. You can override these in your main script.
# ==========================================
# 1. SETUP: Define Models & Hyperparameters
# ==========================================
# You can change numbers here (e.g., n_estimators, hidden_layer_sizes) 
# BEFORE running the analysis.

MODEL_MAP = {
    'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=123),
    'svm': SVR(kernel='rbf', C=100, epsilon=0.1),
    'mr': LinearRegression(),
    'mlpe': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=123),
    'xgboost': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=123)
}




# ==========================================
# 3. TRAINING WRAPPERS
# ==========================================
# train_models, fit_models, movement_analysis, movement_analysis_short_term

def train_models(Wdata, mlds):
    """
    Simulates rminer::mining. 
    Runs 5 runs of 5-fold Cross Validation for specified models.
    Returns metrics and averaged out-of-sample predictions.
    """
    
    # 1. Prepare Data (Assumes 'stage_5' is the target, as per your R code)
    target = 'stage_5'
    if target not in Wdata.columns:
        raise ValueError(f"Target column '{target}' not found in Wdata.")
        
    X = Wdata.drop(columns=[target])
    y = Wdata[target]

    # Output containers
    MM = {}          # To store model objects
    metrics_list = [] # To store MAE, RMSE, etc.
    MM_Data_list = {} # To store Obs vs Pred dataframes

    # 2. Loop through requested models
    for m_name in mlds:
        if m_name not in MODEL_MAP:
            print(f"Warning: Model '{m_name}' not defined in MODEL_MAP. Skipping.")
            continue

        print(f"\nRunning mining :: {m_name}")
        model = MODEL_MAP[m_name]

        # -------------------------------------------------------
        # Part A: Calculate Metrics (MAE, MSE, RMSE, R2)
        # We use Repeated K-Fold (5 splits, 5 repeats) to match R "Runs=5"
        # -------------------------------------------------------
        rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=123)
        
        scoring = {
            'mae': 'neg_mean_absolute_error',
            'mse': 'neg_mean_squared_error',
            'r2': 'r2'
        }
        
        scores = cross_validate(model, X, y, cv=rkf, scoring=scoring)

        # Calculate average metrics (Note: sklearn returns negative errors)
        avg_mae = -np.mean(scores['test_mae'])
        avg_mse = -np.mean(scores['test_mse'])
        avg_rmse = np.sqrt(avg_mse)
        avg_r2 = np.mean(scores['test_r2'])

        metrics_list.append({
            'Model': m_name,
            'MAE': avg_mae,
            'MSE': avg_mse,
            'RMSE': avg_rmse,
            'R2': avg_r2
        })

        # -------------------------------------------------------
        # Part B: Generate MM_Data (Obs vs Pred)
        # To match 'Reduce("+", PP$pred) / length', we average 
        # predictions from 5 different CV runs.
        # -------------------------------------------------------
        
        # Create a placeholder to sum up predictions
        sum_preds = np.zeros(len(y))
        n_runs = 5
        
        for i in range(n_runs):
            # different random_state for each run ensures different splits
            cv_run = KFold(n_splits=5, shuffle=True, random_state=123 + i)
            preds = cross_val_predict(model, X, y, cv=cv_run)
            sum_preds += preds
            
        # Average the predictions
        avg_preds = sum_preds / n_runs
        
        # Create the dataframe
        mm_set = pd.DataFrame({
            'Obs': y.values,
            'Pred': avg_preds,
            'Error': np.abs(y.values - avg_preds),  # Absolute Error (for MAE)
            'Residual': y.values - avg_preds        # Raw Residual (for bias checking)
        })
        
        MM_Data_list[m_name] = mm_set
        MM[m_name] = model # Store the base model definition

    # Compile metrics into a DataFrame
    MET = pd.DataFrame(metrics_list)

    return {'models': MM, 'MET': MET, 'MMData': MM_Data_list}


def fit_models(Wdata, mlds):
    """
    Fits the specified models on the ENTIRE dataset.
    Equivalent to rminer::fit
    """
    target = 'stage_5'
    X = Wdata.drop(columns=[target])
    y = Wdata[target]

    FM = {}

    for m_name in mlds:
        if m_name in MODEL_MAP:
            print(f"\nRunning fit :: {m_name}")
            # Clone ensures we start with a fresh, untrained model
            model = clone(MODEL_MAP[m_name])
            model.fit(X, y)
            FM[m_name] = model
        else:
            print(f"Skipping unknown model: {m_name}")

    return FM



# ==========================================
# 4. PREDICTION WRAPPERS
# ==========================================
# perform_prediction (or perform_prediction_rev), perform_prediction_multiple_short_term

def perform_prediction(model_name, new_dataset_name, intervals=(12, 24, 72, 144), 
                       pred_times=[12], WDir=None, saveresults=None):
    """
    Loads a trained model and performs predictions on a new dataset for specified time horizons.
    Includes fix for SettingWithCopyWarning.
    """
    
    try:
        # 0. Setup Working Directory
        if WDir:
            os.chdir(WDir)
            print(f"Working Directory set to: {WDir}")
        else:
            WDir = os.getcwd()

        # -------------------------------------------------------
        # 1. Load the Saved Model
        # -------------------------------------------------------
        start_time = time.time()
        print("Loading the model...")
        
        # Handle file extension logic
        if not str(model_name).endswith('.pkl'):
            model_path = f"{model_name}.pkl"
        else:
            model_path = model_name
            
        if not os.path.exists(model_path):
             if os.path.exists(f"{model_name}.model"):
                 model_path = f"{model_name}.model"
             else:
                 raise FileNotFoundError(f"Model file not found: {model_path}")
             
        predictor = joblib.load(model_path)
        
        w = predictor.get('window_size', 5)
        models = predictor.get('fit_results', {})
        interval_stats = predictor.get('interval_stats', {})
        filtered_datasets = predictor.get('filtered_datasets', {})
        
        print(f"Model loaded. Window size (w): {w}")
        print(f"Models found: {list(models.keys())}")
        print(f"Time to load model: {time.time() - start_time:.2f} seconds\n")

        # -------------------------------------------------------
        # 2. Load New Dataset
        # -------------------------------------------------------
        start_time = time.time()
        print("Loading new dataset...")
        
        source_dataset = load_data(new_dataset_name)

        if source_dataset is None:
            raise ValueError("Failed to load new dataset.")
        
        if 'Date' in source_dataset.columns:
            source_dataset['Date'] = pd.to_datetime(source_dataset['Date'])

        print(f"Time to load new datasets: {time.time() - start_time:.2f} seconds\n")

        # -------------------------------------------------------
        # 3. Preparation for Fallback Logic
        # -------------------------------------------------------
        raw_data_dates = []
        if isinstance(new_dataset_name, str):
            try:
                raw_df = pd.read_csv(new_dataset_name)
                raw_cols = [str(c).replace('X', '') for c in raw_df.columns if 'PID' not in str(c) and 'Unnamed' not in str(c)]
                raw_data_dates = pd.to_datetime(raw_cols, format='%Y%m%d', errors='coerce')
                raw_data_dates = raw_data_dates[~raw_data_dates.isna()]
            except:
                pass 

        # -------------------------------------------------------
        # 4. Prediction Loop
        # -------------------------------------------------------
        all_predictions = {}
        
        for t in pred_times:
            print(f"Processing for prediction time: {t}")
            loop_start = time.time()
            
            # Initialize variables to None to avoid ReferenceError if logic skips
            new_data = None
            date_data = None
            dates_used = None
            PointID_col = None
            
            # --- SCENARIO A: Fallback / Special Dataset (2 columns: Date, PID1) ---
            if len(source_dataset.columns) == 2 and 'Date' in source_dataset.columns and 'PID1' in source_dataset.columns:
                # print("Applying fallback method for special dataset...")
                
                temp_source = source_dataset.copy()
                if len(raw_data_dates) > 0:
                    temp_source = temp_source[temp_source['Date'].isin(raw_data_dates)]
                
                temp_source = temp_source.sort_values('Date')
                time_diffs = temp_source['Date'].diff().dt.days.values
                
                if len(temp_source) < 4:
                    print(f"Warning: Not enough data for fallback prediction at t={t}")
                    continue

                vals = temp_source['PID1'].values
                ts = time_diffs
                
                # Manual construction of the input window (last 4 steps)
                fallback_data = pd.DataFrame({
                    'PointID': [1], 
                    'Time1': [ts[-3]],      
                    'Time2': [ts[-2]],
                    'Time3': [ts[-1]],
                    'Time4': [t],           
                    'stage_1': [vals[-4]],
                    'stage_2': [vals[-3]],
                    'stage_3': [vals[-2]],
                    'stage_4': [vals[-1]]
                })
                
                new_data = fallback_data.drop(columns=['PointID'])
                PointID_col = fallback_data['PointID']
                date_data = pd.DataFrame() 
                
            # --- SCENARIO B: Standard Processing ---
            else:
                day_available = (source_dataset['Date'].max() - source_dataset['Date'].min()).days
                check_days = np.array([36, 72, 216, 432])
                closest_day_idx = np.argmin(np.abs(day_available - check_days))
                closest_int_idx = np.argmin(np.abs(np.array(intervals) - t))
                final_index = min(closest_int_idx, closest_day_idx)
                selected_interval = intervals[final_index]
                
                try:
                    # w-1 assumes w includes target, so input window is w-1 (e.g. 5-1=4)
                    processed = process_data(source_dataset, selected_interval, w-1)
                    
                    if processed is None:
                        raise ValueError("process_data returned None")
                        
                    comXD = processed['data'].copy()
                    dates_used = processed['dates']
                    
                    PointID_col = comXD['PointID']
                    date_cols = [c for c in comXD.columns if c.startswith("D")]
                    date_data = comXD[date_cols]
                    
                    time_col_name = f"Time{w-1}" 
                    comXD[time_col_name] = t
                    
                    cols_to_drop = ['PointID'] + date_cols
                    new_data = comXD.drop(columns=cols_to_drop)
                    
                    time_cols = [f"Time{i+1}" for i in range(w-1)]
                    stage_cols = [f"stage_{i+1}" for i in range(w-1)]
                    expected_order = time_cols + stage_cols
                    
                    if set(expected_order).issubset(new_data.columns):
                        new_data = new_data[expected_order]
                        
                except Exception as e:
                    print(f"Error processing interval {selected_interval}: {e}")
                    new_data = None
            
            if new_data is None:
                print(f"Skipping prediction time {t} due to data generation failure.")
                continue
                
            print(f"Time to process dataset: {time.time() - loop_start:.2f} seconds")

            # -------------------------------------------------------
            # 5. Perform Predictions
            # -------------------------------------------------------
            pred_start = time.time()
            print(f"Performing prediction for time: {t}")
            
            current_preds_dict = {}
            
            for m_name, model_obj in models.items():
                try:
                    preds = model_obj.predict(new_data)
                    col_name = f"{m_name}_T{t}D"
                    current_preds_dict[col_name] = preds
                except Exception as e:
                    print(f"Prediction error for model {m_name}: {e}")

            pred_df = pd.DataFrame(current_preds_dict)
            
            # -------------------------------------------------------
            # 6. Reconstruct Final DataFrame
            # -------------------------------------------------------
            final_df = pd.concat([
                PointID_col.reset_index(drop=True),
                new_data.reset_index(drop=True),
                date_data.reset_index(drop=True) if not date_data.empty else pd.DataFrame(),
                pred_df.reset_index(drop=True)
            ], axis=1)
            
            all_predictions[f"T{t}D"] = {
                'Predictions': current_preds_dict,
                'DatesUsed': dates_used,
                'NewData': final_df
            }
            
            print(f"Time for predictions: {time.time() - pred_start:.2f} seconds\n")

        # -------------------------------------------------------
        # 7. Post-Processing (Filter Last Entry)
        # -------------------------------------------------------
        filtered_results = {}
        
        for key, res_obj in all_predictions.items():
            df = res_obj['NewData']
            if df.empty: continue
            
            # Group by PointID and take the last row
            last_rows = df.groupby('PointID').tail(1).copy()
            
            # Sort numerically if IDs allow
            try:
                last_rows['_sort_id'] = last_rows['PointID'].astype(str).str.extract(r'(\d+)').astype(float)
                last_rows = last_rows.sort_values('_sort_id').drop(columns=['_sort_id'])
            except:
                last_rows = last_rows.sort_values('PointID')

            filtered_results[key] = last_rows

        # -------------------------------------------------------
        # 8. Save and Return
        # -------------------------------------------------------
        result = {
            'model_name': model_name,
            'pred_times': pred_times,
            'all_predictions': all_predictions,
            'last_four_dates': filtered_results, 
            'window_size': w,
            'model_results': predictor.get('model_results'),
            'fit_results': models,
            'interval_stats': interval_stats,
            'filtered_datasets': filtered_datasets,
            'source_dataset': source_dataset
        }

        if saveresults:
            save_path = os.path.join(WDir, f"{saveresults}.pred.pkl")
            joblib.dump(result, save_path)
            print(f"Results saved as: {save_path}")

        print(f"Total time to save results: {time.time() - start_time:.2f} seconds")
        return result

    except Exception as e:
        print("\n!!! An Error Occurred in perform_prediction !!!")
        import traceback
        traceback.print_exc()
        return None




def risk_processing(pred_name, model_type='mlpe', centers=5, nstart=25, 
                    alpha=0.7, beta=0.3, WDir=None, 
                    original_csv="A13_vertical_2016_2023_clipped.csv", 
                    saveresults=None):
    """
    Python equivalent of the R function 'risk_processing' with enhanced 
    clustering metrics and visualization.
    """
    
    # 0. Set Working Directory
    if WDir:
        print(f"Setting directory to: {WDir}")
        os.chdir(WDir)
        
    # Load Model
    if isinstance(pred_name, str):
        print("Loading the model...")
        predictor = joblib.load(pred_name)
    else:
        predictor = pred_name

    # Extract Data
    try:
        raw_data = predictor['all_predictions']['T12D']['NewData']
    except KeyError:
        raw_data = predictor 

    com12D = raw_data.copy()

    # ───────────────────────────────────────────────────────────────
    # 1. Compute stage differences (DYNAMIC MODEL SELECTION)
    # ───────────────────────────────────────────────────────────────
    print(f"Computing stage differences using model: {model_type}...")
    
    target_col = f"{model_type}_T12D"
    
    if target_col in com12D.columns:
        com12D['pred'] = com12D[target_col]
    else:
        available = [c for c in com12D.columns if "_T12D" in c]
        raise ValueError(f"Model column '{target_col}' not found. Available columns: {available}")
    
    com12D['stagediff12'] = (com12D['stage_2'] - com12D['stage_1']).abs()
    com12D['stagediff23'] = (com12D['stage_3'] - com12D['stage_2']).abs()
    com12D['stagediff34'] = (com12D['stage_4'] - com12D['stage_3']).abs()
    
    com12D['avg_stagediff'] = (com12D['stagediff12'] + com12D['stagediff23'] + com12D['stagediff34']) / 3
    
    diff_cols = ['stagediff12', 'stagediff23', 'stagediff34']
    com12D['sd_stagediff'] = com12D[diff_cols].std(axis=1, ddof=1)
    
    potential_models = [target_col, 'mr_T12D', 'mlpe_T12D', 'xgboost_T12D', 'rf_T12D', 'svm_T12D']
    com12D.drop(columns=potential_models, errors='ignore', inplace=True)

    # ───────────────────────────────────────────────────────────────
    # 2. Prepare PCA
    # ───────────────────────────────────────────────────────────────
    print("Running PCA...")
    features = com12D[['avg_stagediff', 'sd_stagediff', 'stagediff34', 'stage_4']].copy()
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    correlation_features = features.corr()
    
    pca = PCA(n_components=None)
    pca.fit(features_scaled)
    
    # Store variance ratios for plotting later
    var_ratio = pca.explained_variance_ratio_

    summary_pca = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(features.columns))],
        'Var_Explained': var_ratio,
        'Cumulative_Var': np.cumsum(var_ratio)
    })
    
    rotation_pca = pd.DataFrame(pca.components_.T, index=features.columns, columns=summary_pca['PC'])
    
    pca_subset = pca.transform(features_scaled)[:, :2]
    pca_df = pd.DataFrame(pca_subset, columns=['PC1', 'PC2'])

    # ───────────────────────────────────────────────────────────────
    # 3. K-Means clustering on PCA
    # ───────────────────────────────────────────────────────────────
    print(f"Running K-Means (k={centers})...")
    kmeans = KMeans(n_clusters=centers, random_state=123, n_init=nstart)
    kmeans.fit(pca_subset)
    
    # Assign labels
    com12D['pca_cluster'] = kmeans.labels_
    pca_df['cluster'] = kmeans.labels_

    # ───────────────────────────────────────────────────────────────
    # NEW: METRICS & VISUALIZATION
    # ───────────────────────────────────────────────────────────────
    
    # --- C. Print Cluster Centers ---
    print("\n--- Cluster Centers (PC1, PC2) ---")
    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=['PC1', 'PC2'])
    centers_df.index.name = 'Cluster'
    print(centers_df)

    # --- D. Calculate Performance Metrics (WSS / BSS / TSS) ---
    # 1. WSS (Total Within-Cluster Sum of Squares)
    wss = kmeans.inertia_

    # 2. TSS (Total Sum of Squares)
    global_mean = np.mean(pca_subset, axis=0)
    tss = np.sum((pca_subset - global_mean) ** 2)

    # 3. BSS (Between-Cluster Sum of Squares)
    bss = tss - wss

    # 4. Ratio
    ratio = (bss / tss) * 100

    print(f"\nTotal Sum of Squares (TSS): {tss:.2f}")
    print(f"Within-Cluster SS  (WSS): {wss:.2f}")
    print(f"Between-Cluster SS (BSS): {bss:.2f}")
    print(f"BSS / TSS Ratio:          {ratio:.2f}%")

    # --- E. Calculate & Print Cluster Sizes ---
    cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index()
    cluster_props = (cluster_counts / len(kmeans.labels_) * 100).round(2)

    print("\n--- Cluster Sizes (Number of Points) ---")
    print(cluster_counts)
    print("\n--- Cluster Proportions (%) ---")
    print(cluster_props)

    # --- B. Generate Plot with Distinct Colors ---
    # Only generate plot if we can save it or show it
    print("\nGenerating Cluster Plot...")
    plt.figure(figsize=(10, 6))
    
    # Use tab10 for distinct categorical colors
    custom_palette = "tab10" 

    sns.scatterplot(
        data=pca_df, 
        x='PC1', 
        y='PC2', 
        hue='cluster', 
        palette=custom_palette, 
        s=60, 
        edgecolor='black',
        alpha=0.7
    )

    plt.title(f"K-Means Clustering on PCA (k={centers})")
    plt.xlabel(f"PC1: Volatility ({var_ratio[0]*100:.1f}%)")
    plt.ylabel(f"PC2: Displacement State ({var_ratio[1]*100:.1f}%)")
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Determine filename for plot
    plot_filename = "pca_clust_distinct.tiff"
    if saveresults:
        plot_filename = f"{saveresults}_pca_clust.tiff"
        
    plt.savefig(plot_filename, dpi=300, format='tiff', bbox_inches='tight')
    # Note: plt.show() might block execution in automated scripts. 
    # If running interactively, uncomment the line below:
    # plt.show()
    plt.close() # Close figure to free memory
    print(f"Plot saved to: {plot_filename}")

    # ───────────────────────────────────────────────────────────────
    # 4. Dynamic slope class assignment (A–E)
    # ───────────────────────────────────────────────────────────────
    cluster_order = com12D.groupby('pca_cluster')['sd_stagediff'].mean().sort_values().reset_index()
    cluster_order.rename(columns={'sd_stagediff': 'mean_sd'}, inplace=True)
    
    slope_labels = list(string.ascii_uppercase)[:centers]
    cluster_order['slope_class'] = slope_labels
    
    com12D = com12D.merge(cluster_order[['pca_cluster', 'slope_class', 'mean_sd']], 
                          on='pca_cluster', how='left')

    # ───────────────────────────────────────────────────────────────
    # 5. Compute Risk Index dynamically
    # ───────────────────────────────────────────────────────────────
    print("Calculating Risk Index...")
    def rescale(series):
        return (series - series.min()) / (series.max() - series.min())

    score_map = {label: i+1 for i, label in enumerate(slope_labels)}
    com12D['ClassScore'] = com12D['slope_class'].map(score_map)
    com12D['Hazard']     = com12D['stage_4'].abs()
    
    com12D['Hazard_n'] = rescale(com12D['Hazard'])
    com12D['Class_n']  = rescale(com12D['ClassScore'])
    
    com12D['RiskRaw']   = alpha * com12D['Hazard_n'] + beta * com12D['Class_n']
    com12D['RiskIndex'] = rescale(com12D['RiskRaw'])

    # ───────────────────────────────────────────────────────────────
    # 6. Risk category (Quantiles)
    # ───────────────────────────────────────────────────────────────
    breaks_q = np.unique(np.quantile(com12D['RiskIndex'].dropna(), np.linspace(0, 1, 5)))
    labels_risk = ["Low", "Moderate", "High", "Very High"]
    actual_labels = labels_risk[:len(breaks_q)-1]
    
    com12D['RiskCategory'] = pd.cut(com12D['RiskIndex'], bins=breaks_q, labels=actual_labels, include_lowest=True)

    # ───────────────────────────────────────────────────────────────
    # 7. Aggregate risk by PointID
    # ───────────────────────────────────────────────────────────────
    print("Aggregating by PointID...")
    area_risk = com12D.groupby('PointID')['RiskIndex'].max().reset_index()
    area_risk.rename(columns={'RiskIndex': 'MaxRiskIndex'}, inplace=True)
    
    breaks_q_max = np.unique(np.quantile(area_risk['MaxRiskIndex'].dropna(), np.linspace(0, 1, 5)))
    actual_labels_max = labels_risk[:len(breaks_q_max)-1]
    
    area_risk['MaxRiskCategory'] = pd.cut(area_risk['MaxRiskIndex'], bins=breaks_q_max, labels=actual_labels_max, include_lowest=True)

    # ───────────────────────────────────────────────────────────────
    # 8. Spatial join with original coordinates
    # ───────────────────────────────────────────────────────────────
    com12D_sf = None
    area_risk_sf = None
    
    if os.path.exists(original_csv):
        print("Loading original coordinate data...")
        original = pd.read_csv(original_csv)
        original['PointID'] = "PID" + original['fid'].astype(str)
        
        pid_coords = original[['PointID', 'longitude', 'latitude', 'acceleration', 
                               'mean_velocity', 'max_temporal_coherence', 'slope_angle']]
        
        com12D = com12D.merge(pid_coords, on='PointID', how='left')
        area_risk = area_risk.merge(pid_coords, on='PointID', how='left')
        
        print("Creating Spatial Objects...")
        com12D_sf = gpd.GeoDataFrame(com12D, geometry=gpd.points_from_xy(com12D.longitude, com12D.latitude), crs="EPSG:4326")
        area_risk_sf = gpd.GeoDataFrame(area_risk, geometry=gpd.points_from_xy(area_risk.longitude, area_risk.latitude), crs="EPSG:4326")
    else:
        print(f"Warning: '{original_csv}' not found. Skipping spatial join.")

    # ───────────────────────────────────────────────────────────────
    # 9. Save and return results
    # ───────────────────────────────────────────────────────────────
    result = {
        'pred_name': pred_name,
        'model_used': model_type,
        'com12D': com12D,
        'features': features,
        'features_scaled': features_scaled,
        'correlation_features': correlation_features,
        'summary_pca': summary_pca,
        'rotation_pca': rotation_pca,
        'pca_df': pca_df,
        'pca_kmeans_result': kmeans,
        'area_risk': area_risk,
        'com12D_sf': com12D_sf,
        'area_risk_sf': area_risk_sf,
        
        # New Metrics
        'cluster_centers': centers_df,
        'cluster_counts': cluster_counts,
        'metrics': {
            'tss': tss,
            'wss': wss,
            'bss': bss,
            'bss_tss_ratio': ratio
        }
    }
    
    if saveresults:
        filename = f"{saveresults}.pkl"
        print(f"Saving results to {filename}...")
        joblib.dump(result, filename)
        
        if area_risk_sf is not None:
            gpkg_name = f"{saveresults}_spatial.gpkg"
            print(f"Saving spatial layers to {gpkg_name}...")
            com12D_sf.to_file(gpkg_name, layer='com12D_sf', driver="GPKG")
            area_risk_sf.to_file(gpkg_name, layer='area_risk_sf', driver="GPKG")
    
    print("Risk Processing Complete.")
    return result




