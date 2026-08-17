# Executive Brief: Credit Risk and Customer Lifecycle Analytics

## Portfolio Snapshot

- Test customers scored: 3,438
- Selected model: logistic_scorecard
- ROC-AUC: 0.763
- PR-AUC: 0.256
- KS statistic: 0.405
- Recall in riskiest decile: 34.48%
- Policy approval rate: 76.27%
- Approved default rate: 4.20%
- Declined default rate: 18.50%
- Real UCI Default of Credit Card Clients benchmark: gradient_boosting ROC-AUC 0.775, KS 0.421 on 30,000 public records


## Business Interpretation

The project converts raw applicant and customer behavior data into a probability-of-default score, risk bands, approval policy simulation, customer churn readout, and risk-adjusted margin view. This mirrors the analytics charter in banking and fintech roles: acquisition, engagement, retention, risk scorecards, behavioral models, and data-driven portfolio strategy.

## Main Drivers

Top model drivers by permutation importance: credit_score, revolving_utilization, missed_payment_count_6m, employment_tenure_months, acquisition_channel, debt_to_income.

## Monitoring

PSI drift monitoring flagged 0 severe numeric feature shifts between the development window and the latest scoring window. The report is designed for a model governance workflow: monitor, investigate, recalibrate, and document policy changes.

## Policy Optimization

Best constrained threshold from the approval/default frontier:

- PD threshold: 10.00%
- Approval rate: 76.27%
- Approved default rate: 4.20%
- Total risk-adjusted margin: 5,586,754

## Fairness and Compliance Readiness

- Group dimensions checked: 5
- Groups checked: 16
- Adverse-impact watch groups: 1
- Lowest approval-rate ratio: 0.77
