\# Spotlite InSAR Slope Risk Analysis



A modular Python framework for analyzing InSAR displacement data to assess slope stability risks. This tool uses machine learning models (XGBoost, Random Forest, etc.) to predict ground movement and clustering algorithms (K-Means, PCA) to classify risk levels (A-E).



\## 📂 Project Structure



```text

slope-risk-analysis/

├── pyproject.toml       # Dependencies (managed by uv)

├── uv.lock              # Lock file for reproducible builds

├── main.py              # Entry point for the application

├── Dockerfile           # Configuration for Docker deployment

├── .gitignore           # Git exclusion rules

├── data/                # Folder for input CSVs and Model files

└── src/                 # Source Code Package

&nbsp;   ├── \_\_init\_\_.py

&nbsp;   ├── core.py          # Core logic (Training, Prediction, Risk Processing)

&nbsp;   └── utils/

&nbsp;       ├── \_\_init\_\_.py

&nbsp;       └── helpers.py   # Data manipulation tools (Load, Sliding Window)





🚀 Installation \& Setup

This project uses uv for high-speed dependency management.



Prerequisites

1. Install uv (PowerShell):



powershell -c "irm \[https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"



2\. Clone the repository:



git clone \[https://github.com/SteffanDavies/spotlite-insar-sloperisk.git](https://github.com/SteffanDavies/spotlite-insar-sloperisk.git)

cd spotlite-insar-sloperisk



Initialize Environment

Run these commands to set up the environment and install all required libraries:



uv sync





📊 Usage

1\. Prepare Data

Ensure your input files are in the data/ directory:



Input CSV (e.g., ver\_A13.csv)



Trained Model (e.g., Smovement\_I22\_py.model or .pkl)



2\. Run the Analysis

Execute the main script using uv:



uv run main.py



🐳 Running with Docker

You can run the entire analysis in an isolated container without installing Python locally.



1. Build the Image:



docker build -t slope-risk-app .



2\. Run the Container: Map your local data folder to the container so it can read/write files:



docker run -v ${PWD}/data:/app/data slope-risk-app







































































