select
            vintage_month,
            count(*) as customers,
            avg(pd_score) as avg_pd,
            avg(default_12m) as default_rate,
            avg(churn_6m) as churn_rate,
            avg(approve_policy) as approval_rate,
            avg(risk_adjusted_margin) as avg_risk_adjusted_margin
        from scored
        group by 1
        order by 1
