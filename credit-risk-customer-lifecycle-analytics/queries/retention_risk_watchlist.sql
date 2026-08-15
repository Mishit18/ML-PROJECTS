select
            customer_segment,
            acquisition_channel,
            risk_band,
            count(*) as customers,
            avg(churn_6m) as churn_rate,
            avg(default_12m) as default_rate,
            avg(digital_logins_30d) as avg_digital_logins_30d,
            avg(product_count) as avg_product_count,
            avg(risk_adjusted_margin) as avg_risk_adjusted_margin
        from scored
        group by 1, 2, 3
        having count(*) >= 25
        order by churn_rate desc, avg_risk_adjusted_margin desc
