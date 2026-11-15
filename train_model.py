import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# File paths
normal_file = "data/transactions/transactions_0_0.csv"
fraud_file = "data/fraud/transactions_fraud.csv"
accounts_file = "data/accounts/accounts_0_0.csv"

print("Loading transaction data...", flush=True)

columns = ["tx_id", "src_id", "dst_id", "amount", "timestamp"]

normal = pd.read_csv(normal_file, usecols=columns)
fraud = pd.read_csv(fraud_file, usecols=columns)

normal["fraud"] = 0
fraud["fraud"] = 1

data = pd.concat([normal, fraud], ignore_index=True)

print("Total transactions:", len(data))
print("Normal transactions:", (data["fraud"] == 0).sum())
print("Fraud transactions:", (data["fraud"] == 1).sum())

# Timestamp features
data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

data["hour"] = data["timestamp"].dt.hour
data["day_of_week"] = data["timestamp"].dt.dayofweek
data["month"] = data["timestamp"].dt.month
data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

# Account features
print("\nLoading account data...", flush=True)

accounts = pd.read_csv(accounts_file)

source = accounts[
    ["account_id", "balance", "risk_score", "creation_date"]
].rename(columns={
    "account_id": "src_id",
    "balance": "src_balance",
    "risk_score": "src_risk",
    "creation_date": "src_creation_date"
})

destination = accounts[
    ["account_id", "balance", "risk_score", "creation_date"]
].rename(columns={
    "account_id": "dst_id",
    "balance": "dst_balance",
    "risk_score": "dst_risk",
    "creation_date": "dst_creation_date"
})

data = data.merge(source, on="src_id", how="left")
data = data.merge(destination, on="dst_id", how="left")

data["src_creation_date"] = pd.to_datetime(
    data["src_creation_date"], errors="coerce"
)

data["dst_creation_date"] = pd.to_datetime(
    data["dst_creation_date"], errors="coerce"
)

data["src_account_age"] = (
    data["timestamp"] - data["src_creation_date"]
).dt.total_seconds() / 86400

data["dst_account_age"] = (
    data["timestamp"] - data["dst_creation_date"]
).dt.total_seconds() / 86400

# Model features
features = [
    "amount",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "src_balance",
    "src_risk",
    "dst_balance",
    "dst_risk",
    "src_account_age",
    "dst_account_age"
]

X = data[features]
y = data["fraud"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Random Forest model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest model...", flush=True)

model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Fraud"],
    zero_division=0
))

print(
    "Average Precision:",
    round(average_precision_score(y_test, probabilities), 4)
)

# Save model
joblib.dump(model, "fraud_model.joblib")

with open("model_features.json", "w") as file:
    json.dump(features, file, indent=4)

print("\nModel saved successfully!")
print("Training completed!")
