select
            risk_band,
            case
                when annual_income < 400000 then '01_low_income'
                when annual_income < 900000 then '02_mid_income'
                else '03_high_income'
            end as income_band,
            count(*) as customers,
            avg(pd_score) as avg_pd,
            avg(default_12m) as default_rate,
            avg(expected_margin_12m) as avg_expected_margin,
            avg(risk_adjusted_margin) as avg_risk_adjusted_margin
        from scored
        group by 1, 2
        order by 1, 2
