# Credit Risk and Customer Lifecycle Analytics

End-to-end fintech analytics project for data scientist, data analyst, ML, and banking analytics roles. The project builds a reproducible credit portfolio, predicts 12-month default risk, creates underwriting risk bands, evaluates customer churn and risk-adjusted margin, exports SQL KPIs, and documents model governance.

## Why This Project Exists

High-paying fintech and analytics JDs often mention credit risk, behavioral models, customer lifecycle, acquisition, engagement, retention, risk scorecards, dashboards, model monitoring, and business impact. This project is designed to hit those signals without pretending to use private banking data.

## Project Highlights

- Synthetic but realistic portfolio of 25,000 customers with bureau, repayment, utilization, transaction, engagement, channel, churn, and margin fields.
- Temporal train/test split by customer vintage month to avoid random-split leakage.
- Model comparison: scorecard-style logistic regression, gradient boosting, and random forest.
- Evaluation: ROC-AUC, PR-AUC, KS statistic, top-decile recall, precision/recall/F1 at policy threshold.
- Policy simulation: probability of default, risk bands, approval rate, realized default rate, and risk-adjusted margin.
- SQL analytics layer in DuckDB for risk-band, acquisition-channel, segment profitability, and monthly monitoring views.
- Explainability via permutation importance, adverse-action style reason codes, fairness diagnostics, and monitoring via population stability index.
- Real public-data validation on 30,000 customers from UCI Default of Credit Card Clients, plus OpenML German Credit, in addition to the synthetic lifecycle portfolio.
- Calibration comparison across uncalibrated, sigmoid, and isotonic gradient boosting using Brier score, 10-bin ECE, score bands, and bootstrap ROC-AUC intervals.
- Streamlit dashboard for risk policy, lifecycle, fairness, reason codes, and monitoring.
- Reports, plots, tests, and resume bullets generated from one command.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
pytest -q
```

## Repository Structure

```text
credit-risk-customer-lifecycle-analytics/
|-- src/credit_lifecycle/
|   |-- data.py              # Synthetic portfolio generator
|   |-- features.py          # Feature schema and temporal split
|   |-- modeling.py          # Model training, selection, metrics, explainability
|   |-- monitoring.py        # PSI drift and score bands
|   |-- sql_analytics.py     # DuckDB KPI exports
|   `-- reporting.py         # Plots and markdown reports
|-- scripts/run_pipeline.py
|-- tests/test_pipeline_components.py
|-- queries/                 # Generated SQL files
|-- outputs/                 # Generated metrics, plots, scored data
|-- reports/                 # Executive brief, model card, resume bullets
|-- artifacts/               # Generated model artifact and metadata
`-- requirements.txt
```

## Generated Outputs

| Output | Purpose |
|---|---|
| `outputs/model_metrics.csv` | Model comparison across scorecard, boosting, and forest |
| `outputs/scored_customers.csv` | Test-window customers with PD score, risk band, policy approval, margin |
| `outputs/risk_band_policy.csv` | Approval/default/profitability by risk band |
| `outputs/channel_lifecycle.csv` | Acquisition-channel quality and customer lifecycle KPIs |
| `outputs/segment_profitability.csv` | Segment risk, churn, product depth, and margin |
| `outputs/drift_report.csv` | PSI drift by numeric feature |
| `outputs/policy_threshold_frontier.csv` | Approval/default/profitability threshold frontier |
| `outputs/business_policy_simulator.csv` | Decision simulator for approval, default, margin, and declined opportunity cost |
| `queries/acquisition_efficiency.sql` | Channel x region acquisition quality query |
| `queries/cohort_retention_quality.sql` | Vintage cohort retention, default, product-depth, and margin query |
| `queries/collections_prioritization.sql` | High-risk/high-value customer action queue |
| `outputs/fairness_group_metrics.csv` | Approval/default/churn diagnostics by age, income, region, segment, channel |
| `outputs/adverse_action_reason_codes.csv` | Customer-level reason codes for highest-risk cases |
| `outputs/real_openml_german_credit_benchmark.csv` | Real public-data benchmark on OpenML credit-g |
| `outputs/real_uci_calibration_comparison.csv` | Calibration and uncertainty comparison on 30,000 real UCI customers |
| `reports/executive_brief.md` | Business-facing summary |
| `reports/model_card.md` | Model governance documentation |
| `reports/resume_bullets.md` | Ready-to-use resume bullets |
| `reports/INTERVIEW_CASEBOOK.md` | Data analyst / data scientist interview defense and business decision memo |

## Visual Evidence

### Risk-Band Default Separation

![Risk band default rate](outputs/risk_band_default_rate.png)

### Feature Importance

![Feature importance](outputs/feature_importance.png)

### Policy Threshold Frontier

![Policy threshold frontier](outputs/policy_threshold_frontier.png)

### Fairness Watchlist

![Fairness approval ratio watchlist](outputs/fairness_approval_ratio_watchlist.png)

## Resume Bullets

- Built end-to-end credit risk and customer lifecycle analytics pipeline on 25,000 synthetic banking customers, combining bureau, repayment, utilization, acquisition, churn, and margin signals.
- Trained and compared scorecard-style logistic regression, gradient boosting, and random forest models with temporal validation, ROC-AUC, PR-AUC, KS, top-decile recall, and approval-policy simulation.
- Designed DuckDB SQL analytics layer for risk-band policy, approval rate, realized default, churn, segment profitability, acquisition-channel quality, and monthly portfolio monitoring.
- Added permutation explainability, adverse-action style reason codes, fairness diagnostics, PSI drift monitoring, threshold optimization, score-band governance, and risk-adjusted margin simulation to translate model outputs into underwriting and retention actions.
- Validated gradient boosting on 30,000 real UCI credit-card customers, reaching ROC-AUC 0.775 and KS 0.421 with score-band calibration evidence.
- Benchmarked the modeling workflow on OpenML German Credit data, achieving ROC-AUC 0.801 and KS 0.494 on 1,000 public credit-risk records.

## Dashboard

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

## Limitations

- Uses synthetic data for reproducibility and privacy; it is not a production underwriting model.
- The real-data benchmarks are downloaded from their public sources, cached locally, and reported separately from the synthetic lifecycle simulation.
- Real deployment would require bureau data contracts, reject inference, fairness testing, adverse-action reason codes, regulatory review, and live champion/challenger monitoring.
- The UCI benchmark has no observation timestamp, so it uses a stratified holdout rather than claiming invalid out-of-time validation.
