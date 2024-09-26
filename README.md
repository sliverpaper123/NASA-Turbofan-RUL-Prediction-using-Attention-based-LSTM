# NASA Turbofan RUL Prediction using Attention-based LSTM

This project implements a deep learning model for predicting the Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS dataset. The model uses an Attention-based LSTM architecture to capture temporal dependencies in the sensor data and make accurate RUL predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to predict the Remaining Useful Life (RUL) of aircraft turbofan engines based on sensor data. The project uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset, which contains run-to-failure data for a fleet of engines under various operating conditions and fault modes.

Key features of the project:
- Data preprocessing and feature engineering
- Implementation of an Attention-based LSTM model
- Custom loss function for asymmetric penalization
- Early stopping to prevent overfitting
- Evaluation metrics including RMSE, MAE, and R-squared
- Visualization of results

## Dataset

The NASA C-MAPSS dataset consists of four subsets:
- FD001: Single fault mode, single operating condition
- FD002: Single fault mode, six operating conditions
- FD003: Two fault modes, single operating condition
- FD004: Two fault modes, six operating conditions

Each subset contains training data, test data, and true RUL values for the test set.

You can download the dataset from the [NASA Prognostics Center of Excellence Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/nasa-turbofan-rul-prediction.git
   cd nasa-turbofan-rul-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Download the NASA C-MAPSS dataset and place the files in the project directory.

2. Run the main script:
   ```
   python rul_prediction.py
   ```

3. The script will process all four datasets (FD001, FD002, FD003, FD004) and generate results for each.

4. Check the console output for performance metrics and look for the generated plot images in the project directory.

## Model Architecture

The model uses an Attention-based LSTM architecture:
- LSTM layers for capturing temporal dependencies
- Attention mechanism to focus on important time steps
- Fully connected layers for final prediction

The model is trained using a custom loss function that penalizes late predictions more heavily than early predictions.

## Results

The model's performance is evaluated using the following metrics:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared Score

Results for each dataset are printed to the console and saved as text files. Additionally, the following plots are generated for each dataset:
- True vs Predicted RUL scatter plot
- Distribution of prediction errors
