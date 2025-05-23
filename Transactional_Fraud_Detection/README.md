# Transactional Fraud Detection Hackathon Project

## Overview

This project was developed as part of a hackathon on transactional fraud detection, organized in collaboration with **Mastercard** and **SGH (Warsaw School of Economics)**. The challenge involved detecting fraudulent activity in artificial transactional data, simulating real-world banking scenarios.

## Problem Statement

Financial fraud, especially in electronic transactions, poses significant risks to both institutions and customers. The objective was to build a robust machine learning model capable of accurately identifying fraudulent transactions from a provided synthetic dataset.

## Approach

### 1. **Exploratory Data Analysis (EDA)**
- Conducted a thorough EDA to understand transaction patterns, customer behaviors, and the overall distribution of classes.
- **Key Finding:** Identified "localized periods of attacks" that were both temporal (in specific time windows) and spatial (affecting particular locations or regions). This insight was used to inform feature engineering and modeling strategies.

### 2. **Feature Engineering**
- Created and aggregated features capturing temporal, spatial, behavioral, and transactional characteristics.
- Engineered features to highlight suspicious behaviors, such as clustering of high-value transactions, rapid transaction sequences, and region-specific attack waves.
- Added features to capture the aforementioned "localized attack periods" for improved detection capability.

### 3. **Feature Importance Analysis with XGBoost**
- Trained an XGBoost model to evaluate the relative importance of engineered and raw features.
- Used the model’s feature importance rankings to further refine the feature set, focusing on the most predictive attributes.

### 4. **Final Model: LSTM-based Predictor**
- Developed a sequential model using **LSTM (Long Short-Term Memory)** networks to account for temporal dependencies in transaction data.
- The LSTM model was chosen for its effectiveness in sequence modeling, enabling detection of evolving fraudulent behaviors over time and across locations.
- Evaluated model performance using relevant metrics (e.g., ROC-AUC, F1-score) to ensure robust fraud detection.

## Results

- **Extensive EDA** led to actionable insights on fraudulent patterns, including temporal and spatial clustering of attacks.
- **Feature engineering** and **XGBoost analysis** streamlined the feature set for the final model.
- The **LSTM predictor** effectively leveraged transaction sequences, resulting in improved identification of fraudulent activity.

## Technologies Used

- **Python** (pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, TensorFlow/Keras)
- **Jupyter Notebook** for analysis and prototyping
