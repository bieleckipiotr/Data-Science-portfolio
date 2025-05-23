## Dataset Description

# 🔍 Synthetic Card Transactions - Fraud Detection Starter

Welcome to the Mastercard Hackathon starter dataset!

This synthetic dataset simulates credit card transaction activity including both legitimate and fraudulent events. Designed to mimic real-world structures, the dataset encourages entry-level exploration of fraud detection concepts.

## 🔢 What's Inside?

* transactions.json: 500,000 transactions in JSON format with complex structure (location, payment method, session data, etc.)
* users.csv: 20,000 users from Europe, each with detailed demographic and financial information
* merchants.csv: 1,000 European merchants, including behavioral and trust-related features

## 🧠 Suggested Starter Task

Build a machine learning model that classifies transactions as fraudulent (1) or legitimate (0). Your task is to train a model using the provided transaction, user, and merchant data, then submit predictions for a new batch of transactions.

Start by exploring:

* Temporal patterns of fraud
* High-risk merchants and regions
* Cardholder behavior anomalies

### Structure

1. **merchants.csv**

| merchant_id | category | country | trust_score | number_of_alerts_last_6_months | avg_transaction_amount | account_age_months | has_fraud_history |
| ----------- | -------- | ------- | ----------- | ------------------------------ | ---------------------- | ------------------ | ----------------- |
| M0001       | travel   | Austria | 1.0         | 3                              | 97.23                  | 84                 | 0                 |
| M0002       | clothing | Poland  | 0.6897      | 2                              | 142.71                 | 93                 | 1                 |

2. **transactions.json**

```json
{
  "transaction_id": "TX000000",
  "timestamp": "2022-06-17T23:28:00",
  "user_id": "U14804",
  "merchant_id": "M0314",
  "amount": 130.03,
  "channel": "in-store",
  "currency": "EUR",
  "device": "Android",
  "location": {
    "lat": 40.057938,
    "long": 14.959737
  },
  "payment_method": "debit_card",
  "is_international": 1,
  "session_length_seconds": 145,
  "is_first_time_merchant": 0,
  "is_fraud": 0
}
```


3. **users.csv**

| user_id | age | sex    | education   | primary_source_of_income | sum_of_monthly_installments | sum_of_monthly_expenses | country | signup_date | risk_score |
| ------- | --- | ------ | ----------- | ------------------------ | --------------------------- | ----------------------- | ------- | ----------- | ---------- |
| U00001  | 56  | Other  | High School | Employment               | 477.69                      | 243.18                  | Finland | 2021-04-01  | 0.5711     |
| U00002  | 36  | Female | Bachelor    | Business                 | 31.6                        | 737.76                  | France  | 2020-07-07  | 0.7053     |
