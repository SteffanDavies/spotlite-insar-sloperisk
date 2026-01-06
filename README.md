
# Spotlite InSAR Slope Risk Analysis

A modular Python framework for analyzing InSAR displacement data to assess slope stability risks. This tool uses machine learning models (XGBoost, Random Forest, etc.) to predict ground movement and clustering algorithms (K-Means, PCA) to classify risk levels (A-E).

## 📂 Project Structure

```text
slope-risk-analysis/
├── pyproject.toml       # Dependencies (managed by uv)
├── uv.lock              # Lock file for reproducible builds
├── main.py              # Entry point for the application
├── Dockerfile           # Configuration for Docker deployment
├── .gitignore           # Git exclusion rules
├── data/                # Folder for input CSVs and Model files
└── src/                 # Source Code Package
    ├── __init__.py
    ├── core.py          # Core logic (Training, Prediction, Risk Processing)
    └── utils/
        ├── __init__.py
        └── helpers.py   # Data manipulation tools (Load, Sliding Window)

```

## 🚀 Installation & Setup

This project uses **[uv](https://github.com/astral-sh/uv)** for high-speed dependency management.

### Prerequisites

1. **Install uv** (PowerShell):

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

```


2. **Clone the repository:**

```bash
git clone https://github.com/SteffanDavies/spotlite-insar-sloperisk.git

cd spotlite-insar-sloperisk
```

### Initialize Environment

Run these commands to set up the environment and install all required libraries:

```bash
uv sync
```

## 📊 Usage

### 1. Prepare Data

Ensure your input files are in the `data/` directory:

* **Input CSV** (e.g., `ver_A13.csv`)
* **Trained Model** (e.g., `Smovement_I22_py.model` or `.pkl`)

### 2. Run the Analysis

Execute the main script using `uv`:

```bash
uv run main.py
```

## 🐳 Running with Docker

You can run the entire analysis in an isolated container without installing Python locally.

1. **Build the Image:**

```bash
docker build -t slope-risk-app .
```

2. **Run the Container:**

Map your local `data` folder to the container so it can read/write files:
**PowerShell (Windows):**

```powershell
docker run -v ${PWD}/data:/app/data slope-risk-app
```

**Bash (Linux/Mac):**

```bash
docker run -v $(pwd)/data:/app/data slope-risk-app
```
