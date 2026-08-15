from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


SQL_QUERIES = {
    "risk_band_policy": """
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
    """,
    "channel_lifecycle": """
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
    """,
    "segment_profitability": """
        select
            customer_segment,
            count(*) as customers,
            avg(pd_score) as avg_pd,
            avg(default_12m) as default_rate,
            avg(churn_6m) as churn_rate,
            avg(product_count) as avg_products,
            avg(risk_adjusted_margin) as avg_risk_adjusted_margin
        from scored
        group by 1
        order by avg_risk_adjusted_margin desc
    """,
    "monthly_monitoring": """
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
    """,
    "cohort_retention_quality": """
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
    """,
    "ltv_default_segments": """
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
    """,
    "acquisition_efficiency": """
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
    """,
    "retention_risk_watchlist": """
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
    """,
    "pricing_risk_grid": """
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
    """,
    "collections_prioritization": """
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
    """,
}


def export_sql_files(query_dir: Path) -> None:
    query_dir.mkdir(parents=True, exist_ok=True)
    for name, sql in SQL_QUERIES.items():
        (query_dir / f"{name}.sql").write_text(sql.strip() + "\n", encoding="utf-8")


def run_sql_analytics(scored: pd.DataFrame, output_dir: Path, query_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_sql_files(query_dir)
    con = duckdb.connect()
    con.register("scored", scored)
    outputs = {}
    for name, sql in SQL_QUERIES.items():
        df = con.execute(sql).df()
        outputs[name] = df
        df.to_csv(output_dir / f"{name}.csv", index=False)
    con.close()
    return outputs
