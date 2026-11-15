
import pandas as pd

normal_file = "data/transactions/transactions_0_0.csv"
fraud_file = "data/fraud/transactions_fraud.csv"

print("Reading fraud transaction IDs...")

fraud = pd.read_csv(fraud_file, usecols=["tx_id"])
fraud_ids = set(fraud["tx_id"])

print("Total fraud records:", len(fraud_ids))

print("\nChecking main transaction file...")

normal_count = 0
overlap_ids = set()

for chunk in pd.read_csv(
    normal_file,
    usecols=["tx_id"],
    chunksize=50000
):
    normal_count += len(chunk)
    overlap_ids.update(set(chunk["tx_id"]) & fraud_ids)

print("Total main transaction records:", normal_count)
print("Fraud IDs also found in main file:", len(overlap_ids))

if overlap_ids:
    print("Sample overlapping IDs:", list(overlap_ids)[:10])
else:
    print("No overlapping transaction IDs found.")
