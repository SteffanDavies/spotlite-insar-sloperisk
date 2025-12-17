```markdown
# Slope Stability Risk Assessment Framework

## Introduction
The **Slope Stability Risk Assessment Framework** is a Python-based analytical toolset designed to evaluate and categorize landslide risk in geological monitoring areas. By leveraging Machine Learning (ML) predictions of ground displacement stages, this framework quantifies risk through a spatial-temporal analysis of volatility and displacement magnitude.

This tool automates the transition from raw displacement predictions to actionable risk intelligence, producing spatial datasets (GIS-ready), statistical metrics, and visualization plots to assist geotechnical engineers in decision-making.

## Methodology
The risk assessment pipeline follows a two-stage process:

### 1. Displacement Prediction
The system utilizes pre-trained ML models (e.g., XGBoost, MLPE, Random Forest) to predict the future displacement stage (1–4) of monitoring points based on historical movement data.
* **Input**: Historical velocity, acceleration, and temporal coherence data.
* **Output**: Predicted displacement stages for defined time intervals (e.g., T12, T24).

### 2. Risk Processing (PCA-KMeans Clustering)
Raw predictions are transformed into a **Risk Index** using a multi-step statistical approach:
1.  **Volatility Calculation**: Computes the variance and average magnitude of stage transitions (e.g., shifting from Stage 1 to Stage 3).
2.  **Dimensionality Reduction (PCA)**: Principal Component Analysis reduces the feature space into two primary components:
    * **PC1 (Volatility)**: How erratic the movement is.
    * **PC2 (Displacement State)**: The severity of the current movement.
3.  **Behavioral Clustering**: K-Means clustering groups monitoring points into distinct **Slope Classes** (A, B, C, D, E) based on their PCA signatures.
4.  **Risk Index Computation**: A normalized score (0.0 to 1.0) is calculated dynamically:
    $$Risk = \alpha \times Hazard + \beta \times Vulnerability$$
    * *Hazard*: Normalized magnitude of the predicted stage (Stage 4 = High Hazard).
    * *Vulnerability*: Normalized score of the Slope Class (Class E = High Historical Volatility).
    * Default weights: $\alpha=0.7$ (Hazard), $\beta=0.3$ (Behavior).

## Project Structure
```text
.
├── models/                  # Directory containing trained .model files
├── data/                    # Input CSV files (spatial coordinates, sensor data)
├── output/                  # Generated results (.pkl, .gpkg, .tiff)
├── scripts/
│   ├── main.py              # Main execution script
│   └── risk_utils.py        # Contains risk_processing() and helper functions
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies

```

## Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn geopandas matplotlib seaborn joblib xgboost

```

*Note: `xgboost` is required only if using XGBoost models for prediction.*

## Usage

### Step 1: Generate Predictions

Run the prediction module to generate raw displacement forecasts.

```python
from risk_utils import perform_prediction

# Run prediction
predict_movement = perform_prediction(
    model_name="models/Smovement_I22_py.model",
    new_dataset_name="data/ver_A13.csv",
    intervals=[12],
    pred_times=[12],
    WDir="./",
    saveresults="output/risk_predictions"
)

```

### Step 2: Process Risk

Run the risk processing module to calculate the Risk Index, generate clusters, and export GIS layers.

```python
from risk_utils import risk_processing

# Run risk analysis
results = risk_processing(
    pred_name="output/risk_predictions.pkl",  # Load output from Step 1
    model_type="xgboost",
    centers=5,               # Number of Slope Classes (Clusters)
    alpha=0.7,               # Weight for Magnitude
    beta=0.3,                # Weight for Volatility
    original_csv="data/A13_vertical.csv", # Required for spatial coordinates
    saveresults="output/final_risk_analysis"
)

```

## Outputs

The framework generates three primary outputs in the `saveresults` location:

1. **Results Dictionary (`.pkl`)**:
* Contains the full Pandas DataFrame, PCA coordinates, Cluster Centers, and Performance Metrics (WSS, TSS, BSS).


2. **Spatial Layer (`.gpkg`)**:
* A GeoPackage containing two layers:
* `com12D_sf`: Individual points with Risk Categories.
* `area_risk_sf`: Aggregated maximum risk per PointID.




3. **Cluster Plot (`.tiff`)**:
* High-resolution scatter plot visualizing the K-Means clusters on the PC1/PC2 axis.



## License

This project is licensed under the spotlite License - see the [LICENSE]() file for details.

## Contact

**Author**: [Dominic/Spotlite]

**Email**: [dominicowusuansah8@gmail.com]

**Version**: 0

```

```

