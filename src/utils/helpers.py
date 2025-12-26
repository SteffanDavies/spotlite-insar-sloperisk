# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 11:45:15 2025

@author: domin
"""

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
import joblib
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import geopandas as gpd
import string



# ==========================================
# 1. UTILITIES
# ==========================================
# load_data, extract_data, extract_data_short_term, sliding_window, 
# filter_by_time_intervals, calculate_statistics, rename_and_clean_columns

# --- FUNCTION 1: LOAD DATA ---
def load_data(input_file):
    """
    Loads data, transposes it, fixes dates, and interpolates missing time steps.
    """
    try:
        if isinstance(input_file, pd.DataFrame):
            print("Loading data from provided dataframe")
            data = input_file.copy()
        elif isinstance(input_file, str):
            print(f"Loading data from file: {input_file}")
            # IMPORTANT: Using default index loading based on our fix
            data = pd.read_csv(input_file) 
        else:
            raise ValueError("Error: Input must be either a file path (str) or a dataframe.")

        # Transpose and cleanup
        actual_workdata = data.T 
        actual_workdata.columns = [f"PID{i+1}" for i in range(len(actual_workdata.columns))]
        
        actual_workdata.index.name = 'Date'
        actual_workdata = actual_workdata.reset_index()
        
        # Clean Date strings
        actual_workdata['Date'] = actual_workdata['Date'].astype(str).str.replace("X", "", regex=False)
        if actual_workdata['Date'].iloc[0].startswith("Unnamed"):
             actual_workdata = actual_workdata.iloc[1:]

        actual_workdata['Date'] = pd.to_datetime(actual_workdata['Date'], format='%Y%m%d')
        actual_workdata = actual_workdata.dropna()

        # Interpolation (The clean 12-day grid method)
        actual_workdata = actual_workdata.set_index('Date')
        full_date_range = pd.date_range(start=actual_workdata.index.min(), 
                                      end=actual_workdata.index.max(), freq='12D')
        
        interpolated_data = actual_workdata.reindex(full_date_range).interpolate(method='linear')
        actual_workdata = interpolated_data.reset_index().rename(columns={'index': 'Date'})

        return actual_workdata

    except Exception as e:
        print(f"Error in load_data: {e}")
        return None


# ---------------------------------------------------------
# 1. Data Splitting Functions
# ---------------------------------------------------------

def extract_data(data, num_splits):
    """
    Dynamically splits the dataframe into 'num_splits' subsets.
    """
    print(f"Splitting source data into {num_splits} parts...")
    
    # Numpy array_split is the standard equivalent to R's chunking logic
    chunks = np.array_split(data, num_splits)
    
    # Create dictionary keys W1, W2, etc.
    split_data = {f"W{i+1}": chunk for i, chunk in enumerate(chunks)}
    
    return split_data



def sliding_window(values, dates, point_id, window_size=5):
    """
    Creates a DataFrame of sliding windows for values and dates.
    """
    # Ensure inputs are numpy arrays
    values = np.array(values)
    dates = pd.to_datetime(dates).values 
    
    # Create Windows using NumPy
    val_windows = sliding_window_view(values, window_size)
    date_windows = sliding_window_view(dates, window_size)
    
    # Calculate time differences (days)
    time_diffs = np.diff(date_windows, axis=1).astype('timedelta64[D]').astype(int)
    
    # Construct Result Dictionary
    data = {'PointID': point_id}
    
    for i in range(window_size):
        data[f'stage_{i+1}'] = val_windows[:, i]
    for i in range(window_size):
        data[f'D{i+1}'] = date_windows[:, i]
    for i in range(window_size - 1):
        data[f'Time{i+1}'] = time_diffs[:, i]
        
    return pd.DataFrame(data)


def filter_by_time_intervals(data, time_interval):
    """
    Filters rows where all Time columns (Time1...Time4) match the specified interval.
    """
    # Select the columns of interest
    time_cols = ['Time1', 'Time2', 'Time3', 'Time4']
    
    # Check if ALL columns in a row equal the time_interval
    # .all(axis=1) ensures every column matches for that row
    mask = (data[time_cols] == time_interval).all(axis=1)
    
    return data[mask].copy()


def calculate_statistics(data):
    """
    Calculates the standard deviation of the 'Error' column.
    """
    # ddof=1 is required to match R's default 'sd()' function (Sample Standard Deviation)
    return data['Error'].std(ddof=1)


def rename_and_clean_columns(data, model_name):
    """
    Selects model-specific columns and renames them to generic 'Obs', 'Pred', 'Error'.
    Dynamically handles any model name (mr, mlpe, rf, etc.) without if/else chains.
    """
    # 1. Define the base columns we always want to keep
    base_cols = [
        'Time1', 'Time2', 'Time3', 'Time4',
        'stage_1', 'stage_2', 'stage_3', 'stage_4', 'stage_5'
    ]
    
    # 2. Identify model-specific columns (e.g., 'mr.Obs', 'mlpe.Pred')
    # We find columns that start with "model_name."
    prefix = f"{model_name}."
    model_cols = [col for col in data.columns if col.startswith(prefix)]
    
    if not model_cols:
        print(f"Warning: No columns found for model '{model_name}'")
        return data

    # 3. Select columns (intersection of what we want vs what exists)
    cols_to_keep = [c for c in base_cols + model_cols if c in data.columns]
    cleaned_data = data[cols_to_keep].copy()

    # 4. Create renaming map dynamically
    # e.g., {'mr.Obs': 'Obs', 'mr.Pred': 'Pred'}
    rename_map = {col: col.replace(prefix, "") for col in model_cols}
    
    cleaned_data.rename(columns=rename_map, inplace=True)
    
    return cleaned_data


# ==========================================
# 2. DATA PROCESSING
# ==========================================
# process_data, process_data_rev, com_datasets, com_datasets_short_term
# (Exclude process_data_v1 if redundant)


def process_data(Wx, interval, w):
    """
    Filters data by specific interval and applies sliding window.
    """
    print(f"Processing data for interval: {interval} days...")
    
    # 1. Prepare Data
    Wx = Wx.copy()
    Wx['Date'] = pd.to_datetime(Wx['Date'])
    
    # 2. Generate sequence of dates based on interval
    start_date = Wx['Date'].min()
    end_date = Wx['Date'].max()
    new_dates = pd.date_range(start=start_date, end=end_date, freq=f'{interval}D')
    
    # 3. Subset data to match these dates
    dataset = Wx[Wx['Date'].isin(new_dates)].copy()
    
    if dataset.empty:
        print(f"Warning: No data found matching interval {interval}.")
        return None

    # 4. Process Columns
    dates = dataset['Date']
    measurement_cols = [col for col in dataset.columns if col != 'Date']
    
    results_list = []
    for col in measurement_cols:
        values = dataset[col].values
        df_window = sliding_window(values, dates, col, w)
        results_list.append(df_window)
    
    # Combine results
    if not results_list:
        return None
        
    new_dataset = pd.concat(results_list, ignore_index=True)
    
    # 5. Formatting Dates
    date_cols = [f'D{i+1}' for i in range(w)]
    for col in date_cols:
        new_dataset[col] = new_dataset[col].dt.strftime('%Y%m%d')
        
    # 6. Reorder Columns
    time_cols = [f'Time{i+1}' for i in range(w-1)]
    stage_cols = [f'stage_{i+1}' for i in range(w)]
    final_order = ['PointID'] + time_cols + stage_cols + date_cols
    
    return {'data': new_dataset[final_order], 'dates': new_dates}



def com_datasets(processed_data_list):
    """
    Combines a list of processed results into Balanced and Unbalanced datasets.
    """
    # Extract just the DataFrame ('data') from the result dictionary
    valid_datasets = [item['data'] for item in processed_data_list if item is not None]

    if not valid_datasets:
        raise ValueError("No valid datasets to combine!")

    # 1. Unbalanced: Stack everything
    unbalanced_data = pd.concat(valid_datasets, ignore_index=True)

    # 2. Balanced: Sample to match the smallest dataset size
    min_length = min(len(df) for df in valid_datasets)
    
    balanced_list = []
    for df in valid_datasets:
        # Note: Python's random state 123 is NOT the same as R's set.seed(123)
        sampled = df.sample(n=min_length, random_state=123)
        balanced_list.append(sampled)
        
    balanced_data = pd.concat(balanced_list, ignore_index=True)

    return {
        'unbalanced_data': unbalanced_data, 
        'balanced_data': balanced_data
    }

