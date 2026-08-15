select
            customer_id,
            risk_band,
            pd_score,
            expected_margin_12m,
            risk_adjusted_margin,
            revolving_utilization,
            missed_payment_count_6m,
            payment_to_min_ratio,
            case
                when pd_score >= 0.20 and expected_margin_12m >= 2500 then 'high-touch save or line review'
                when pd_score >= 0.20 then 'automated risk intervention'
                when churn_6m = 1 and risk_adjusted_margin > 2500 then 'retention offer'
                else 'standard monitoring'
            end as recommended_action
        from scored
        order by pd_score desc, expected_margin_12m desc
        limit 250
