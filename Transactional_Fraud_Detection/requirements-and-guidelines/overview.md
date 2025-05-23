## Overview

🎓 Welcome to the Mastercard Fraud Detection Hackathon!
Every time you tap your card, make an online purchase, or use a digital wallet, millions of data points are working behind the scenes to make that transaction secure. But not all transactions are what they seem—fraudulent activity can slip through, costing billions globally and affecting real people.

Our goal is to step into the shoes of a Mastercard data scientist and take on the challenge of building a machine learning model that can spot fraud before it happens.

We won't be dealing with real cardholder data. We’ve got prepared a fully synthetic dataset that mirrors real-world transaction behavior, including patterns typical of fraudsters. Our main goal is to analyze, model, and predict which transactions are suspicious, using Python and other favorite data science tools.

### Description

🚀 Goal
The goal of this challenge is to build a machine learning model that can accurately predict fraudulent transactions based on synthetic transactional data.

🎯 Task Overview
You are provided with synthetic credit card transaction data that includes both legitimate and fraudulent transactions. Your task is to develop a classification model that predicts whether a transaction is fraudulent based on various features such as transaction amount, merchant risk, cardholder demographics, and more.

🛠️ What We Need to Do
Data Exploration: Analyze the provided dataset (transactions.json) to understand the features and target label.

Feature Engineering: Create additional features that may improve the model's predictive power.

Model Building: Train a classification model to predict the target variable (is_fraud), where:

* 0 represents a legitimate transaction.
* 1 represents a fraudulent transaction.

Model Evaluation: Use appropriate metrics (Primary matric is accuracy, others lik precission, recall, ROC AUC, classification report are nice to have) to evaluate the model's performance.

Optional: Implement cross-validation and hyperparameter tuning to optimize the model.

### Deliverables

link

keyboard_arrow_up

As a complete solution we would consider a bundle of:

* Well-formatted and well-documented python script (file in format .py or .ipynb).
* Short (1-2 pages) presentation in .pdf format with selected classification metrices (mandatory: Accuracy, optional: Roc Auc, Recall, Precision, classification report), conclusions, brief explanation of approach, techniques and interim steps in modelling process (splitting dataset, encoding, normalization, sampling, etc.).
* Optional: You can also include in the presentation and implementation additional tools (e.g., voter models, external data sources, dimensionality reduction) to enrich the solution.

### Evaluation

link

keyboard_arrow_up

Your submission will be evaluated based on:

* Quality of code (struture of code, use functions and wrappers, optimization) and code documentation (general descriptions, docstrings for functions/packages/modules, type annotations).
* Model Quality: How well your model predicts fraudulent transactions.
* Innovation in Feature Engineering: The creativity and usefulness of features you engineer.
* Presentation: How well you document your approach, code, and insights. What assumptions did You make and how could these be utilized in real-life scenarios.
