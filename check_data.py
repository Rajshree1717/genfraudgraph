
import pandas as pd

normal_file = "data/transactions/transactions_0_0.csv"
fraud_file = "data/fraud/transactions_fraud.csv"

print("Checking normal transaction data...")

normal = pd.read_csv(normal_file, nrows=5)
print("Normal columns:")
print(normal.columns.tolist())
print("\nNormal sample:")
print(normal[["tx_id", "src_id", "dst_id", "amount"]])

print("\nChecking fraud transaction data...")

fraud = pd.read_csv(fraud_file, nrows=5)
print("Fraud columns:")
print(fraud.columns.tolist())
print("\nFraud sample:")
print(fraud[["tx_id", "src_id", "dst_id", "amount"]])

print("\nFraud transaction IDs:")
print(fraud["tx_id"].tolist())
