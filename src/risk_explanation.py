
def explain_customer_risk(customer):

    reasons = []
    recommendations = []

    if customer["usage_change_pct"] < -30:
        reasons.append(
            "Product usage declined significantly."
        )
        recommendations.append(
            "Schedule a product adoption review."
        )

    if customer["logins_last_30d"] < 8:
        reasons.append(
            "Customer login activity is low."
        )
        recommendations.append(
            "Contact the customer to understand adoption barriers."
        )

    if customer["unresolved_tickets"] >= 2:
        reasons.append(
            "Multiple support tickets remain unresolved."
        )
        recommendations.append(
            "Escalate outstanding support tickets."
        )

    if customer["avg_csat"] < 3.5:
        reasons.append(
            "Customer satisfaction is below target."
        )
        recommendations.append(
            "Arrange a customer experience review."
        )

    if customer["days_since_csm_contact"] > 30:
        reasons.append(
            "Customer has not been contacted recently."
        )
        recommendations.append(
            "Schedule a proactive CSM check-in."
        )

    if customer["days_to_renewal"] <= 30:
        reasons.append(
            "Subscription renewal is approaching."
        )
        recommendations.append(
            "Start a renewal conversation immediately."
        )

    if not recommendations:
        reasons.append(
            "No major risk indicators detected."
        )
        recommendations.append(
            "Continue regular account monitoring."
        )

    return {
        "risk_reasons": reasons,
        "recommended_actions": recommendations
    }


def enrich_customer_records(df):

    results = df.apply(
        explain_customer_risk,
        axis=1
    )

    import pandas as pd

    result_df = pd.DataFrame(results.tolist())

    return pd.concat(
        [df.reset_index(drop=True),
         result_df.reset_index(drop=True)],
        axis=1
    )
