select
            customer_segment,
            risk_band,
            count(*) as customers,
            avg(expected_margin_12m) as avg_expected_margin,
            avg(risk_adjusted_margin) as avg_risk_adjusted_margin,
            avg(pd_score) as avg_predicted_default,
            avg(default_12m) as realized_default_rate,
            avg(churn_6m) as churn_rate
        from scored
        group by 1, 2
        having count(*) >= 20
        order by avg_risk_adjusted_margin desc
