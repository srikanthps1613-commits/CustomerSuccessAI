import os
import numpy as np
import pandas as pd

# Reproducible dataset
np.random.seed(42)

TOTAL_CUSTOMERS = 500

industries = [
    "Technology", "Healthcare", "Finance",
    "Retail", "Education", "Manufacturing"
]

plans = ["Basic", "Professional", "Enterprise"]
contracts = ["Monthly", "Quarterly", "Annual"]

records = []

for i in range(1, TOTAL_CUSTOMERS + 1):

    tenure = np.random.randint(1, 61)

    plan = np.random.choice(
        plans, p=[0.35, 0.45, 0.20]
    )

    contract = np.random.choice(
        contracts, p=[0.45, 0.20, 0.35]
    )

    mrr = {
        "Basic": np.random.randint(49, 150),
        "Professional": np.random.randint(150, 500),
        "Enterprise": np.random.randint(500, 2000)
    }[plan]

    logins = max(
        0, int(np.random.normal(20, 9))
    )

    usage_change = round(
        np.random.normal(-5, 25), 2
    )

    total_features = 12

    active_features = np.random.randint(
        2, total_features + 1
    )

    tickets = np.random.poisson(2)

    unresolved = np.random.randint(
        0, min(tickets, 4) + 1
    )

    csat = round(
        np.clip(np.random.normal(4.0, 0.8), 1, 5), 2
    )

    nps = np.random.randint(-100, 101)

    days_since_contact = np.random.randint(0, 91)

    renewal_days = np.random.randint(1, 181)

    # Churn probability based on customer signals
    risk = 0.08

    if usage_change < -30:
        risk += 0.20

    if logins < 8:
        risk += 0.15

    if unresolved >= 2:
        risk += 0.15

    if csat < 3:
        risk += 0.15

    if days_since_contact > 45:
        risk += 0.10

    if contract == "Monthly":
        risk += 0.10

    if tenure < 6:
        risk += 0.08

    if nps < 0:
        risk += 0.08

    risk = min(risk, 0.90)

    churned = int(np.random.random() < risk)

    records.append({
        "customer_id": f"CUST-{i:04d}",
        "company_name": f"Company {i:04d}",
        "industry": np.random.choice(industries),
        "plan": plan,
        "contract_type": contract,
        "monthly_recurring_revenue": mrr,
        "tenure_months": tenure,
        "logins_last_30d": logins,
        "usage_change_pct": usage_change,
        "active_features": active_features,
        "total_features": total_features,
        "support_tickets_30d": tickets,
        "unresolved_tickets": unresolved,
        "avg_csat": csat,
        "nps": nps,
        "days_since_csm_contact": days_since_contact,
        "days_to_renewal": renewal_days,
        "churned": churned
    })


# Create dataset
df = pd.DataFrame(records)

os.makedirs("data", exist_ok=True)

output_path = "data/customers.csv"

df.to_csv(output_path, index=False)

print("=" * 50)
print("CUSTOMERSUCCESS AI - DATA GENERATION")
print("=" * 50)

print(f"\nTotal customers: {len(df)}")

print(
    f"Churned customers: {df['churned'].sum()}"
)

print(
    f"Retained customers: {(df['churned'] == 0).sum()}"
)

print(
    f"Churn rate: {df['churned'].mean() * 100:.2f}%"
)

print(
    f"Total MRR: ${df['monthly_recurring_revenue'].sum():,.2f}"
)

print(f"\nDataset saved to: {output_path}")

print("\nFirst 5 customer records:")
print(df.head().to_string(index=False))

print("\nData generation completed successfully!")
