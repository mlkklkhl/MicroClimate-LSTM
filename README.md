# MicroClimate Project using LSTM

## Objective
This project implements a multivariate LSTM model for predicting indoor heat index values using Bayesian optimization to find the best hyperparameters. The model uses various environmental sensors and outdoor conditions as input features.

```
model_lstm.py
Data Preprocessing and Filtering: Z-score
Cross-Validation Approach: Simple train-test split (67% train, 33% test)
Evaluation Metrics: RMSE
Optimization Strategy: Single optimization run

model_lstm_optim.py
Data Preprocessing and Filtering: filtering by time of day (6:00-18:00) and window/door states
Cross-Validation Approach: Time Series Cross-Validation with multiple splits (5 or 10)
Evaluation Metrics: RMSE, MAE, and R²
Optimization Strategy: Multiple optimization runs with different configurations
```

## Key Features
- Multivariate Time Series Prediction: Uses multiple input features to predict heat index values
- Bayesian Optimization: Automatically finds optimal hyperparameters using Gaussian Process optimization
- GPU Support: Utilizes CUDA for accelerated training when available
- Outlier Removal: Filters out outliers using Z-score method
- Comprehensive Evaluation: Calculates RMSE for both training and test sets

## Input Features (19 total)
- Grid9_Heat_Index
- Outdoor temperature and humidity sensors (Out1-4)
- Outdoor current sensors (Outdoor1-2)
- Curtain_State
- Thermostat
- Air velocity sensors (P1-P4)
## Target Feature
- Grid1_Heat_Index (indoor heat index to predict)

## Model Architecture
The model uses a Multivariate LSTM with the following components:
- LSTM layers with configurable hidden size and number of layers
- Fully connected output layer
- Mean Squared Error loss function
- Adam optimizer

## Hyperparameter Search Space
The Bayesian optimization searches for the best combination of:
- Lookback window (1-7 time steps)
- Hidden size (25-100 units)
- Number of LSTM layers (1-5)
- Learning rate (0.001-0.05, log scale)
- Batch size (32-128)
- Training epochs (50-300)

## Output Files
The optimization process generates:
- CSV files: Results for each trial and final best parameters
- Visualizations: Prediction plots for each trial
- Performance metrics: RMSE values for training and test sets
  
## Dependencies
```
Python 3.x
PyTorch
scikit-optimize (skopt)
scikit-learn
pandas
numpy
matplotlib
scipy
```
