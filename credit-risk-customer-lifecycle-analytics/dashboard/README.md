# Dashboard Notes

The current project exports dashboard-ready CSV files under `outputs/`:

- `risk_band_policy.csv`
- `channel_lifecycle.csv`
- `segment_profitability.csv`
- `monthly_monitoring.csv`
- `drift_report.csv`

Run the Streamlit dashboard:

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

The dashboard includes risk-band policy, lifecycle KPIs, fairness watchlists, adverse-action style reason codes, drift monitoring, and the real OpenML benchmark.
