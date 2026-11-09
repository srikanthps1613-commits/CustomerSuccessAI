import json
import os
import joblib
import pandas as pd

from src.health_score import add_health_scores
from src.churn_prediction import (
    train_churn_model,
    predict_churn
)
from src.risk_explanation import (
    enrich_customer_records
)


# Directories
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


print("=" * 60)
print("CUSTOMERSUCCESS AI - INTELLIGENCE ENGINE")
print("=" * 60)


# 1. Load customer data
print("\n[1] Loading customer dataset...")

df = pd.read_csv("data/customers.csv")

print(f"Loaded {len(df)} customers.")


# 2. Calculate health scores
print("\n[2] Calculating customer health scores...")

df = add_health_scores(df)

print("Health scoring completed.")


# 3. Train churn model
print("\n[3] Training Random Forest churn model...")

model, metrics, decision_threshold = train_churn_model(df)

print("Model training completed.")


# 4. Generate churn probabilities
print("\n[4] Predicting churn probabilities...")

df["churn_probability"] = predict_churn(
    model, df
)

# Model classification threshold
df["predicted_churn"] = (
    df["churn_probability"] >= decision_threshold
).astype(int)

# Business risk bands remain separate
df["churn_risk"] = df["churn_probability"].apply(
    lambda probability:
        "High Risk" if probability >= 0.70
        else "Medium Risk" if probability >= 0.40
        else "Low Risk"
)

print(
    f"Optimized decision threshold: {decision_threshold:.4f}"
)

print("Churn predictions completed.")


# 5. Generate explanations
print("\n[5] Generating customer risk explanations...")

df = enrich_customer_records(df)


# Save trained model
joblib.dump(
    model,
    "models/churn_model.pkl"
)

# Save optimized threshold
joblib.dump(
    decision_threshold,
    "models/decision_threshold.pkl"
)

# 6. Save customer intelligence report
df.to_csv(
    "reports/customer_intelligence_report.csv",
    index=False
)

# Save evaluation metrics
with open(
    "reports/model_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("Model, threshold, customer report and metrics saved.")


# 7. Display model evaluation
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

for metric, value in metrics.items():

    print(f"\n{metric.upper()}:")

    if isinstance(value, (float, int)):
        print(f"{value:.4f}")
    else:
        print(value)


# 8. Portfolio summary
print("\n" + "=" * 60)
print("CUSTOMER SUCCESS PORTFOLIO SUMMARY")
print("=" * 60)

print(
    "\nHigh-risk accounts:",
    (df["churn_risk"] == "High Risk").sum()
)

print(
    "Medium-risk accounts:",
    (df["churn_risk"] == "Medium Risk").sum()
)

print(
    "Low-risk accounts:",
    (df["churn_risk"] == "Low Risk").sum()
)

print(
    "At-risk health accounts:",
    (df["health_status"] == "At Risk").sum()
)

print(
    "Total MRR:",
    f"${df['monthly_recurring_revenue'].sum():,.2f}"
)

print("\nTop 10 highest-risk customers:")

print(
    df.sort_values(
        "churn_probability",
        ascending=False
    )[
        [
            "customer_id",
            "company_name",
            "health_score",
            "churn_probability",
            "churn_risk",
            "monthly_recurring_revenue"
        ]
    ].head(10).to_string(index=False)
)

print("\nINTELLIGENCE ENGINE EXECUTION COMPLETED!")
