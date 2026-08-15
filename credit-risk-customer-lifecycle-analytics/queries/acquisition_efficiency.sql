select
            acquisition_channel,
            region,
            count(*) as customers,
            avg(approve_policy) as approval_rate,
            avg(default_12m) as default_rate,
            avg(churn_6m) as churn_rate,
            avg(expected_margin_12m) as avg_expected_margin,
            avg(risk_adjusted_margin) as avg_risk_adjusted_margin
        from scored
        group by 1, 2
        having count(*) >= 20
        order by avg_risk_adjusted_margin desc
