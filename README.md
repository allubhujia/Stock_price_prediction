# 📈 Stock Price Prediction with Random Forest

A comprehensive machine learning project for predicting stock prices using Random Forest Regression with advanced technical indicators and feature engineering.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Option 1: Full Pipeline](#option-1-full-pipeline-recommended)
  - [Option 2: Individual Scripts](#option-2-individual-scripts)
- [How It Works](#how-it-works)
- [Technical Features](#technical-features)
- [Model Configuration](#model-configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **Random Forest Regression** model to predict stock prices based on historical data and technical indicators. The system automatically downloads stock data, engineers relevant features, trains a robust model, and provides detailed performance metrics and visualizations.

### Key Highlights:
- ✅ 30+ technical indicators and features
- ✅ Random Forest model with optimized hyperparameters
- ✅ Comprehensive evaluation metrics (RMSE, MAE, R²)
- ✅ Beautiful visualizations for analysis

---

## 🚀 Features

### Data Processing
- Multiple technical indicators (Moving Averages, Momentum, Volatility)
- Time-based features (Day of Week, Month, Quarter)
- Lag features for temporal patterns
- Data normalization for optimal model performance

### Model Training
- Random Forest Regressor with configurable parameters
- Train-test split with time-series awareness
- Model persistence for reuse
- Feature importance analysis

### Evaluation & Visualization
- Multiple performance metrics
- Actual vs. Predicted price plots
- Feature importance charts
- Prediction error analysis

---

## 📁 Project Structure

```
stock-price-prediction/
│
├── 📂 data/                          # Data storage directory
│   ├── raw/                          # Raw stock data from Yahoo Finance
│   │   └── NSE-TATAGLOBAL11.csv
│   └── processed/                    # Preprocessed training/testing data
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── 📂 notebooks/                     # Jupyter notebooks for exploration
│   └── EDA.ipynb         # Data exploration and visualization
│
├── 📂 src/                           # Source code
│   ├── config/                       # Configuration settings
│   │   └── settings.py              # Global settings and parameters
│   │
│   ├── data_preparation/             # Data processing modules
│   │   ├── load_data.py             # Download and load stock data
│   │   ├── features.py              # Feature engineering
│   │   ├── preprocess.py            # Data preprocessing and splitting
│   │   └── save_processed.py        # Save/load processed data
│   │
│   ├── models/                       # Model training and inference
│   │   ├── train_regression.py      # Train Random Forest model
│   │   ├── evaluate.py              # Model evaluation and metrics
│   │   ├── infer.py                 # Make predictions
│   │   └── model.pkl                # Saved trained model
│   │
│   └── pipelines/                    # End-to-end pipeline
│       └── stock_pipeline.py        # Complete automated pipeline
│
├── 📂 scripts/                       # Standalone executable scripts
│   ├── run_preprocess.py            # Run data preprocessing only
│   ├── run_training.py              # Run model training only
│   ├── run_evaluation.py            # Run model evaluation only
│   └── run_inference.py             # Run inference only
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

### 📌 Key Directories Explained:

- **`data/`**: Stores all raw and processed data files
- **`notebooks/`**: Interactive Jupyter notebooks for data exploration
- **`src/`**: Core source code organized by functionality
  - **`config/`**: Centralized configuration management
  - **`data_preparation/`**: All data processing logic
  - **`models/`**: Model training, evaluation, and inference
  - **`pipelines/`**: Automated end-to-end workflow
- **`scripts/`**: Standalone scripts for running individual pipeline steps (useful for debugging or custom workflows)

> **Note**: The `scripts/` folder contains standalone executables for running individual components. These are **separate** from the main pipeline and are useful when you want to run only specific steps (e.g., only preprocessing or only training) without executing the entire pipeline.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/allubhujia/stock-price-prediction.git
cd stock-price-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies Include:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Data visualization
- `jupyter` - Interactive notebooks

---

## 💻 Usage

### Option 1: Full Pipeline (Recommended)

Run the complete end-to-end pipeline that handles everything automatically:

```bash
python src/pipelines/stock_pipeline.py
```

**This will:**
2. ✅ Engineer 30+ technical features
3. ✅ Split and normalize data
4. ✅ Train Random Forest model
5. ✅ Evaluate model performance
6. ✅ Generate visualizations

**Expected Output:**
```
======================================================================                                                                                                                             
STOCK RETURN PREDICTION PIPELINE - RANDOM FOREST
======================================================================

[STEP 1/6] Loading stock data...
Using existing stock data...
Loaded 1235 records

[STEP 2/6] Creating technical features...

[STEP 3/6] Preparing and splitting data...
Features shape: (1216, 30)
Target shape: (1216,)
Feature columns: ['Last', 'Turnover (Lacs)', 'log_return', 'MA_5', 'STD_5', 'Volume_MA_5', 'MA_10', 'STD_10', 'Volume_MA_10', 'MA_20', 'STD_20', 'Volume_MA_20', 'Momentum_5', 'Momentum_10', 'Volatility_5', 'Volatility_10', 'HL_Spread', 'HL_Pct', 'VWAP', 'Close_Lag_1', 'Volume_Lag_1', 'Close_Lag_2', 'Volume_Lag_2', 'Close_Lag_3', 'Volume_Lag_3', 'Close_Lag_5', 'Volume_Lag_5', 'DayOfWeek', 'Month', 'Quarter']

Training set size: 972
Testing set size: 244

[STEP 4/6] Saving processed data...

Processed data saved:
X_train: /mnt/c/VIT_CHENNAI/Computer_science/Machine Learning/Personal Projects/stock-price-prediction/data/processed/X_train.npy
X_test: /mnt/c/VIT_CHENNAI/Computer_science/Machine Learning/Personal Projects/stock-price-prediction/data/processed/X_test.npy
y_train: /mnt/c/VIT_CHENNAI/Computer_science/Machine Learning/Personal Projects/stock-price-prediction/data/processed/y_train.npy
y_test: /mnt/c/VIT_CHENNAI/Computer_science/Machine Learning/Personal Projects/stock-price-prediction/data/processed/y_test.npy

[STEP 5/6] Training Random Forest model...

Training Random Forest model...
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    0.2s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    0.4s finished
Training completed!

Model saved to /mnt/c/VIT_CHENNAI/Computer_science/Machine Learning/Personal Projects/stock-price-prediction/src/models/model.pkl

[STEP 6/6] Evaluating model performance...

Evaluating model...
[Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.0s
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.1s finished

==================================================
Model Performance Metrics:
==================================================
RMSE: 0.003653
MAE: 0.001209
R² Score: 0.9732
==================================================


======================================================================
PIPELINE COMPLETED SUCCESSFULLY
======================================================================
```

---

### Option 2: Individual Scripts

Run individual components separately for more control or debugging:

#### Step 1: Preprocess Data
```bash
python scripts/run_preprocess.py
```
Downloads stock data, creates features, and saves processed arrays.

#### Step 2: Train Model
```bash
python scripts/run_training.py
```
Loads processed data and trains the Random Forest model.

#### Step 3: Evaluate Model
```bash
python scripts/run_evaluation.py
```
Evaluates model performance and generates visualizations.

#### Step 4: Run Inference
```bash
python scripts/run_inference.py
```
Makes predictions on test data and displays results.

> **💡 Tip**: The scripts in the `scripts/` folder are standalone and independent of the pipeline. Use them when you need granular control over individual steps or for debugging specific components.

---

## 🔍 How It Works

### 1. Data Collection
- Downloads historical stock data From NSETATAGLOBAL
- Stores raw Open, High, Low, Close, Volume data

### 2. Feature Engineering
Creates **30+ technical indicators**:

**Price-Based Features:**
- Returns and Log Returns
- Price Momentum (5-day, 10-day)
- High-Low Spread and Percentage

**Moving Averages:**
- Simple Moving Averages (5, 10, 20 days)
- Volume Moving Averages

**Volatility Indicators:**
- Rolling Standard Deviation (5, 10 days)
- Return Volatility

**Lag Features:**
- Previous close prices (1, 2, 3, 5 days)
- Previous volumes

**Time Features:**
- Day of Week
- Month
- Quarter

**Advanced Indicators:**
- VWAP (Volume Weighted Average Price)

### 3. Model Training
- **Algorithm**: Random Forest Regressor
- **Why Random Forest?**
  - Handles non-linear relationships
  - Robust to outliers
  - Provides feature importance
  - No need for feature scaling (but we do it anyway for consistency)

**Model Parameters:**
```python
n_estimators=100          # Number of trees
max_depth=10             # Maximum tree depth
min_samples_split=5      # Minimum samples to split
min_samples_leaf=2       # Minimum samples per leaf
random_state=42          # For reproducibility
```

### 4. Evaluation
Measures performance using:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

---

## 🎨 Technical Features

### Generated Features (35 total):

| Category | Features |
|----------|----------|
| **Returns** | Returns, Log_Returns |
| **Moving Averages** | MA_5, MA_10, MA_20 |
| **Volatility** | STD_5, STD_10, Volatility_5, Volatility_10 |
| **Volume** | Volume_MA_5, Volume_MA_10, Volume_MA_20 |
| **Momentum** | Momentum_5, Momentum_10 |
| **Spread** | HL_Spread, HL_Pct |
| **VWAP** | VWAP |
| **Lag Features** | Close_Lag_1 to Close_Lag_5, Volume_Lag_1 to Volume_Lag_4 |
| **Time Features** | DayOfWeek, Month, Quarter |

---

## ⚙️ Model Configuration

Edit `src/config/settings.py` to customize:

```python
# Stock Selection
START_DATE = '2013-10-08'    # Start date for data
END_DATE = '2018-10-08'      # End date for data

# Data Split
TEST_SIZE = 0.2              # 20% for testing
RANDOM_STATE = 42            # For reproducibility

# Feature Engineering
WINDOW_SIZES = [5, 10, 20]   # Moving average windows

# Random Forest Parameters
RF_N_ESTIMATORS = 100        # Number of trees
RF_MAX_DEPTH = 10            # Maximum depth
RF_MIN_SAMPLES_SPLIT = 5     # Min samples to split
RF_MIN_SAMPLES_LEAF = 2      # Min samples per leaf
```

---

## 📊 Results

### Sample Output:

```
==================================================
Model Performance Metrics:
==================================================
RMSE: 0.003653
MAE: 0.001209
R² Score: 0.9732
==================================================
```

### Visualizations Generated:

1. **Actual vs Predicted Prices**
   - Line plot comparing true and predicted values
   - Helps visualize model accuracy over time

2. **Prediction Scatter Plot**
   - Shows correlation between actual and predicted
   - Diagonal line represents perfect predictions

3. **Feature Importance**
   - Bar chart showing top contributing features
   - Helps understand which indicators matter most

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions:
- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Implement other algorithms (AdaBoost, XGBoost)
- Add real-time prediction capabilities
- Create web interface
- Add more visualization options

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Predicting! 📈🚀**
