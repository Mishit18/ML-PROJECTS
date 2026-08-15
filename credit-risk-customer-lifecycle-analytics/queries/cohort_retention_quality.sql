select
            vintage_month,
            count(*) as customers,
            avg(case when churn_6m = 0 then 1 else 0 end) as retained_6m_rate,
            avg(default_12m) as default_rate,
            avg(product_count) as avg_product_count,
            avg(digital_logins_30d) as avg_digital_logins_30d,
            avg(risk_adjusted_margin) as avg_risk_adjusted_margin
        from scored
        group by 1
        order by 1
