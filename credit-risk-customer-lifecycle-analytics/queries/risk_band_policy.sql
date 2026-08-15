select
            risk_band,
            count(*) as customers,
            avg(default_12m) as realized_default_rate,
            avg(pd_score) as avg_predicted_default,
            avg(approve_policy) as approval_rate,
            avg(expected_margin_12m) as avg_expected_margin,
            avg(risk_adjusted_margin) as avg_risk_adjusted_margin
        from scored
        group by 1
        order by 1
