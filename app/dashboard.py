import ast
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CustomerSuccess AI",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = BASE_DIR / "reports" / "customer_intelligence_report.csv"
METRICS_PATH = BASE_DIR / "reports" / "model_metrics.json"


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(REPORT_PATH)

    # Convert stored list columns back into Python lists
    list_columns = ["health_reasons", "risk_reasons", "recommended_actions"]

    for column in list_columns:
        if column in df.columns:
            df[column] = df[column].apply(
                lambda value: ast.literal_eval(value)
                if isinstance(value, str) and value.startswith("[")
                else []
            )

    return df


@st.cache_data
def load_model_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

try:
    df = load_data()
    metrics = load_model_metrics()

except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.info(
        "Run `python train_model.py` from your project folder "
        "to generate the customer report and model metrics."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 CustomerSuccess AI")
st.subheader("AI-Powered Customer Health, Churn Risk & Retention Intelligence")

st.info(
    "Portfolio data is synthetically generated for this project. "
    "Model results demonstrate a machine-learning workflow and "
    "should not be interpreted as validated real-world churn performance."
)


# --------------------------------------------------
# MODEL PERFORMANCE & EVALUATION
# --------------------------------------------------

with st.expander("🧪 Model Performance & Evaluation", expanded=True):

    st.subheader("Held-out Test Results")

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Accuracy",
        f"{metrics['accuracy']:.1%}"
    )

    metric_columns[1].metric(
        "Precision",
        f"{metrics['precision']:.1%}"
    )

    metric_columns[2].metric(
        "Recall",
        f"{metrics['recall']:.1%}"
    )

    metric_columns[3].metric(
        "F2 Score",
        f"{metrics['f2_score']:.1%}"
    )

    st.divider()

    detail_columns = st.columns(3)

    detail_columns[0].metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.4f}"
    )

    detail_columns[1].metric(
        "Average Precision",
        f"{metrics['average_precision']:.4f}"
    )

    detail_columns[2].metric(
        "Decision Threshold",
        f"{metrics['decision_threshold']:.4f}"
    )

    st.caption(
        "The decision threshold is selected using validation data "
        "and evaluated on a separate held-out test set."
    )

    st.write("**Dataset split**")

    split_columns = st.columns(3)

    split_columns[0].metric(
        "Training Records",
        metrics["train_size"]
    )

    split_columns[1].metric(
        "Validation Records",
        metrics["validation_size"]
    )

    split_columns[2].metric(
        "Test Records",
        metrics["test_size"]
    )

    st.write("**Confusion Matrix**")

    confusion_matrix = metrics["confusion_matrix"]

    confusion_figure = px.imshow(
        confusion_matrix,
        text_auto=True,
        x=["Predicted Retained", "Predicted Churn"],
        y=["Actual Retained", "Actual Churn"],
        labels={
            "x": "Prediction",
            "y": "Actual",
            "color": "Customers"
        },
        color_continuous_scale="Blues"
    )

    confusion_figure.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(
        confusion_figure,
        use_container_width=True
    )

    if "classification_report" in metrics:
        st.write("**Classification Report**")
        st.code(metrics["classification_report"])


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Portfolio Filters")

risk_options = ["All"] + sorted(
    df["churn_risk"].dropna().unique().tolist()
)

selected_risk = st.sidebar.selectbox(
    "Churn Risk",
    risk_options
)

plan_options = ["All"] + sorted(
    df["plan"].dropna().unique().tolist()
)

selected_plan = st.sidebar.selectbox(
    "Subscription Plan",
    plan_options
)

industry_options = ["All"] + sorted(
    df["industry"].dropna().unique().tolist()
)

selected_industry = st.sidebar.selectbox(
    "Industry",
    industry_options
)


# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------

filtered_df = df.copy()

if selected_risk != "All":
    filtered_df = filtered_df[
        filtered_df["churn_risk"] == selected_risk
    ]

if selected_plan != "All":
    filtered_df = filtered_df[
        filtered_df["plan"] == selected_plan
    ]

if selected_industry != "All":
    filtered_df = filtered_df[
        filtered_df["industry"] == selected_industry
    ]


# --------------------------------------------------
# PORTFOLIO KPIs
# --------------------------------------------------

st.divider()
st.header("Customer Portfolio Overview")

total_customers = len(filtered_df)

high_risk_customers = len(
    filtered_df[filtered_df["churn_risk"] == "High Risk"]
)

at_risk_customers = len(
    filtered_df[filtered_df["health_status"] == "At Risk"]
)

total_mrr = filtered_df["monthly_recurring_revenue"].sum()

high_risk_mrr = filtered_df.loc[
    filtered_df["churn_risk"] == "High Risk",
    "monthly_recurring_revenue"
].sum()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

kpi2.metric(
    "High Churn Risk",
    f"{high_risk_customers:,}"
)

kpi3.metric(
    "Poor Health Accounts",
    f"{at_risk_customers:,}"
)

kpi4.metric(
    "Total MRR",
    f"${total_mrr:,.0f}"
)

kpi5.metric(
    "High-Risk MRR",
    f"${high_risk_mrr:,.0f}"
)


# --------------------------------------------------
# EMPTY FILTER RESULT HANDLING
# --------------------------------------------------

if filtered_df.empty:
    st.warning(
        "No customers match the selected filters. "
        "Change your filters to view the portfolio."
    )
    st.stop()


# --------------------------------------------------
# CHURN RISK DISTRIBUTION
# --------------------------------------------------

st.divider()
st.header("Customer Risk Analytics")

chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Churn Risk Distribution")

    risk_counts = (
        filtered_df["churn_risk"]
        .value_counts()
        .reindex(
            ["Low Risk", "Medium Risk", "High Risk"],
            fill_value=0
        )
        .reset_index()
    )

    risk_counts.columns = ["Churn Risk", "Customers"]

    risk_figure = px.pie(
        risk_counts,
        names="Churn Risk",
        values="Customers",
        hole=0.45
    )

    st.plotly_chart(
        risk_figure,
        use_container_width=True
    )


# --------------------------------------------------
# CUSTOMER HEALTH DISTRIBUTION
# --------------------------------------------------

with chart2:
    st.subheader("Customer Health Distribution")

    health_counts = (
        filtered_df["health_status"]
        .value_counts()
        .reset_index()
    )

    health_counts.columns = ["Health Status", "Customers"]

    health_figure = px.bar(
        health_counts,
        x="Health Status",
        y="Customers",
        text="Customers"
    )

    health_figure.update_layout(
        xaxis_title="Health Status",
        yaxis_title="Number of Customers"
    )

    st.plotly_chart(
        health_figure,
        use_container_width=True
    )


# --------------------------------------------------
# CUSTOMER PORTFOLIO TABLE
# --------------------------------------------------

st.divider()
st.header("Customer Intelligence Report")

display_columns = [
    "customer_id",
    "company_name",
    "industry",
    "plan",
    "monthly_recurring_revenue",
    "health_score",
    "health_status",
    "churn_probability",
    "churn_risk"
]

available_columns = [
    column for column in display_columns
    if column in filtered_df.columns
]

portfolio_df = filtered_df[available_columns].copy()

if "churn_probability" in portfolio_df.columns:
    portfolio_df["churn_probability"] = (
        portfolio_df["churn_probability"].apply(
            lambda value: f"{value:.1%}"
        )
    )

if "monthly_recurring_revenue" in portfolio_df.columns:
    portfolio_df["monthly_recurring_revenue"] = (
        portfolio_df["monthly_recurring_revenue"].apply(
            lambda value: f"${value:,.2f}"
        )
    )

st.dataframe(
    portfolio_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# CUSTOMER RETENTION ACTION CENTER
# --------------------------------------------------

st.divider()
st.header("🎯 Customer Retention Action Center")

retention_df = filtered_df.copy()

# Calculate estimated revenue exposure
retention_df["estimated_revenue_exposure"] = (
    retention_df["monthly_recurring_revenue"]
    * retention_df["churn_probability"]
)

# Assign CSM follow-up priority
def assign_csm_priority(customer):

    days = customer["days_to_renewal"]
    risk = customer["churn_risk"]

    if risk == "High Risk" and days <= 30:
        return "Critical"

    elif risk == "High Risk" or days <= 14:
        return "High"

    elif risk == "Medium Risk" or days <= 30:
        return "Medium"

    else:
        return "Monitor"


retention_df["csm_priority"] = retention_df.apply(
    assign_csm_priority,
    axis=1
)

# Priority sorting
priority_order = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Monitor": 3
}

retention_df["priority_order"] = (
    retention_df["csm_priority"].map(priority_order)
)

retention_df = retention_df.sort_values(
    by=["priority_order", "estimated_revenue_exposure"],
    ascending=[True, False]
)

# --------------------------------------------------
# RETENTION KPIs
# --------------------------------------------------

critical_accounts = len(
    retention_df[retention_df["csm_priority"] == "Critical"]
)

high_priority_accounts = len(
    retention_df[retention_df["csm_priority"] == "High"]
)

renewal_accounts = len(
    retention_df[retention_df["days_to_renewal"] <= 30]
)

estimated_exposure = retention_df[
    "estimated_revenue_exposure"
].sum()

r1, r2, r3, r4 = st.columns(4)

r1.metric("Critical Accounts", critical_accounts)
r2.metric("High Priority Accounts", high_priority_accounts)
r3.metric("Renewals Within 30 Days", renewal_accounts)
r4.metric(
    "Estimated Revenue Exposure",
    f"${estimated_exposure:,.0f}"
)

st.caption(
    "Revenue exposure is an illustrative estimate calculated as "
    "monthly recurring revenue × model-estimated churn probability. "
    "It is not actual or guaranteed lost revenue."
)

# --------------------------------------------------
# PRIORITY FILTER
# --------------------------------------------------

priority_filter = st.selectbox(
    "Filter by CSM Follow-up Priority",
    ["All", "Critical", "High", "Medium", "Monitor"],
    key="retention_priority_filter"
)

action_df = retention_df.copy()

if priority_filter != "All":
    action_df = action_df[
        action_df["csm_priority"] == priority_filter
    ]

# --------------------------------------------------
# RETENTION ACTION TABLE
# --------------------------------------------------

st.subheader("Prioritized Customer Follow-ups")

action_columns = [
    "customer_id",
    "company_name",
    "csm_priority",
    "churn_risk",
    "churn_probability",
    "monthly_recurring_revenue",
    "estimated_revenue_exposure",
    "days_to_renewal",
    "recommended_actions"
]

available_action_columns = [
    column for column in action_columns
    if column in action_df.columns
]

action_display = action_df[
    available_action_columns
].copy()

# Format recommendation lists for display and CSV
if "recommended_actions" in action_display.columns:
    action_display["recommended_actions"] = (
        action_display["recommended_actions"].apply(
            lambda actions: "; ".join(actions)
            if isinstance(actions, list)
            else str(actions)
        )
    )

st.dataframe(
    action_display,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# DOWNLOAD RETENTION REPORT
# --------------------------------------------------

csv_data = action_display.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Retention Action Report",
    data=csv_data,
    file_name="customer_retention_action_report.csv",
    mime="text/csv"
)

st.caption(
    "Priority labels are rule-based recommendations for CSM review. "
    "They do not represent automated customer outreach."
)

# --------------------------------------------------
# CUSTOMER ACCOUNT DETAILS
# --------------------------------------------------

st.divider()
st.header("Customer Account Intelligence")

customer_options = filtered_df[
    ["customer_id", "company_name"]
].copy()

customer_options["display_name"] = (
    customer_options["customer_id"]
    + " - "
    + customer_options["company_name"]
)

selected_customer = st.selectbox(
    "Select a customer account",
    customer_options["display_name"].tolist()
)

selected_customer_id = selected_customer.split(" - ", 1)[0]

customer = filtered_df[
    filtered_df["customer_id"] == selected_customer_id
].iloc[0]


# --------------------------------------------------
# CUSTOMER PROFILE
# --------------------------------------------------

st.subheader(
    f"{customer['company_name']} ({customer['customer_id']})"
)

profile1, profile2, profile3, profile4 = st.columns(4)

profile1.metric(
    "Monthly Revenue",
    f"${customer['monthly_recurring_revenue']:,.2f}"
)

profile2.metric(
    "Health Score",
    f"{customer['health_score']}/100"
)

profile3.metric(
    "Churn Probability",
    f"{customer['churn_probability']:.1%}"
)

profile4.metric(
    "Days to Renewal",
    f"{customer['days_to_renewal']}"
)


# --------------------------------------------------
# CUSTOMER DETAILS
# --------------------------------------------------

details1, details2 = st.columns(2)

with details1:
    st.markdown("**Account Information**")

    st.write(f"Industry: {customer['industry']}")
    st.write(f"Subscription Plan: {customer['plan']}")
    st.write(f"Contract Type: {customer['contract_type']}")
    st.write(f"Customer Tenure: {customer['tenure_months']} months")
    st.write(f"Health Status: {customer['health_status']}")
    st.write(f"Churn Risk: {customer['churn_risk']}")

with details2:
    st.markdown("**Product Engagement**")

    st.write(f"Logins in Last 30 Days: {customer['logins_last_30d']}")
    st.write(f"Usage Change: {customer['usage_change_pct']}%")
    st.write(
        f"Feature Adoption: "
        f"{customer['active_features']}/{customer['total_features']}"
    )
    st.write(f"Support Tickets: {customer['support_tickets_30d']}")
    st.write(f"Unresolved Tickets: {customer['unresolved_tickets']}")
    st.write(f"Average CSAT: {customer['avg_csat']}")
    st.write(f"NPS: {customer['nps']}")
    st.write(
        f"Days Since Last CSM Contact: "
        f"{customer['days_since_csm_contact']}"
    )


# --------------------------------------------------
# RISK EXPLANATION
# --------------------------------------------------

st.divider()

risk_column, action_column = st.columns(2)

with risk_column:
    st.subheader("⚠️ Identified Risk Indicators")

    risk_reasons = customer.get("risk_reasons", [])

    if risk_reasons:
        for reason in risk_reasons:
            st.warning(reason)
    else:
        st.success("No major risk indicators identified.")


with action_column:
    st.subheader("🎯 Recommended CSM Actions")

    recommended_actions = customer.get(
        "recommended_actions",
        []
    )

    if recommended_actions:
        for action in recommended_actions:
            st.info(action)
    else:
        st.success("No immediate action recommended.")


# --------------------------------------------------
# HEALTH SCORE EXPLANATION
# --------------------------------------------------

st.divider()
st.subheader("Customer Health Score Factors")

health_reasons = customer.get("health_reasons", [])

if health_reasons:
    for reason in health_reasons:
        st.write(f"• {reason}")
else:
    st.success("No health score deductions identified.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CustomerSuccess AI | Customer Health Monitoring, "
    "Churn Prediction & Retention Intelligence | "
    "Portfolio demonstration using synthetic data"
)
