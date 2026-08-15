select
            acquisition_channel,
            count(*) as customers,
            avg(default_12m) as default_rate,
            avg(churn_6m) as churn_rate,
            avg(avg_monthly_spend) as avg_monthly_spend,
            avg(risk_adjusted_margin) as avg_risk_adjusted_margin
        from scored
        group by 1
        order by avg_risk_adjusted_margin desc
