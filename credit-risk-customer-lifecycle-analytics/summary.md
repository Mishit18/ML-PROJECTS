# Project Summary

This project adds the missing fintech/data-science signal for campus roles that mention credit risk, customer lifecycle, risk scorecards, behavioral modeling, acquisition, engagement, retention, and portfolio analytics.

## Current Results

- Customers generated: 25,000
- Test-window customers scored: 3,438
- Selected model: logistic scorecard
- ROC-AUC: 0.763
- PR-AUC: 0.256
- KS statistic: 0.405
- Recall in riskiest decile: 34.48%
- Simulated approval rate: 76.27%
- Approved default rate: 4.20%
- Declined default rate: 18.50%
- Real OpenML German Credit benchmark: ROC-AUC 0.801, PR-AUC 0.645, KS 0.494 on 1,000 public records
- Fairness diagnostics: 5 dimensions and 16 groups checked; 1 adverse-impact watch group flagged
- Policy frontier: constrained threshold selected at 10.00% PD, producing total risk-adjusted margin of 5,586,754 on the test window

## Why It Is Resume-Relevant

The project is stronger than a generic Kaggle classifier because it goes from modeling to business decisioning:

- Temporal validation instead of random split
- Score bands and approval policy
- SQL KPI layer
- Churn and lifecycle analysis
- Risk-adjusted margin
- Explainability
- Drift monitoring
- Model card and executive brief
- Real public-data benchmark
- Adverse-action style reason codes
- Fairness/compliance watchlist
- Threshold optimization frontier
- Streamlit dashboard

## Best Resume Placement

- Data Analytics / Data Scientist resume: primary project
- ML resume: optional fintech applied ML project if replacing a weaker or unfinished item
- Strategy/Ops resume: optional venture/fintech analytics support point

It should not be used on the Quant Trader resume unless a fintech/risk role explicitly asks for credit risk.
