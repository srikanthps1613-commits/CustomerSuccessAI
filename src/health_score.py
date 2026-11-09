
import pandas as pd


def calculate_health_score(customer):
    """
    Calculate customer health score between 0 and 100.
    Higher score = healthier customer.
    """

    score = 100
    reasons = []

    # Product usage
    if customer["logins_last_30d"] < 8:
        score -= 20
        reasons.append("Low product login activity")

    # Usage trend
    if customer["usage_change_pct"] < -30:
        score -= 20
        reasons.append("Significant decline in product usage")
    elif customer["usage_change_pct"] < -15:
        score -= 10
        reasons.append("Declining product usage")

    # Feature adoption
    adoption = (
        customer["active_features"] /
        customer["total_features"]
    )

    if adoption < 0.3:
        score -= 15
        reasons.append("Low feature adoption")
    elif adoption < 0.5:
        score -= 8
        reasons.append("Limited feature adoption")

    # Support experience
    if customer["unresolved_tickets"] >= 3:
        score -= 15
        reasons.append("Multiple unresolved support tickets")
    elif customer["unresolved_tickets"] >= 1:
        score -= 7
        reasons.append("Customer has unresolved support issues")

    # Customer satisfaction
    if customer["avg_csat"] < 3:
        score -= 15
        reasons.append("Low customer satisfaction")
    elif customer["avg_csat"] < 3.5:
        score -= 8
        reasons.append("Customer satisfaction needs attention")

    # CSM engagement
    if customer["days_since_csm_contact"] > 60:
        score -= 15
        reasons.append("No recent CSM engagement")
    elif customer["days_since_csm_contact"] > 30:
        score -= 8
        reasons.append("Customer engagement is overdue")

    # NPS
    if customer["nps"] < 0:
        score -= 10
        reasons.append("Negative Net Promoter Score")

    score = max(0, min(100, score))

    if score >= 80:
        status = "Healthy"
    elif score >= 60:
        status = "Needs Attention"
    else:
        status = "At Risk"

    return {
        "health_score": score,
        "health_status": status,
        "health_reasons": reasons
    }


def add_health_scores(df):

    results = df.apply(
        calculate_health_score,
        axis=1
    )

    result_df = pd.DataFrame(results.tolist())

    return pd.concat(
        [df.reset_index(drop=True),
         result_df.reset_index(drop=True)],
        axis=1
    )