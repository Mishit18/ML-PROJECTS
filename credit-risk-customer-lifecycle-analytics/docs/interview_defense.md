# Interview Defense: Credit Risk and Customer Lifecycle Analytics

## 30-Second Pitch

I built a credit-risk and customer-lifecycle analytics project that goes beyond a classifier. It scores default risk, creates approval bands, analyzes churn and acquisition quality, optimizes approval thresholds, checks fairness-style group diagnostics, creates adverse-action reason codes, monitors feature drift, and validates the workflow on a real public OpenML credit dataset.

## Why This Is Stronger Than a Generic ML Project

- It connects model scores to underwriting policy.
- It reports ROC-AUC, PR-AUC, KS, top-decile recall, approval rate, realized default, and risk-adjusted margin.
- It uses a temporal split for the synthetic lifecycle portfolio rather than a random split.
- It includes a real public-data benchmark on OpenML German Credit.
- It has explainability, reason codes, fairness diagnostics, drift monitoring, SQL KPIs, and a dashboard.

## Key Numbers

| Area | Result |
|---|---:|
| Synthetic portfolio size | 25,000 customers |
| Test-window customers scored | 3,438 |
| Synthetic lifecycle ROC-AUC | 0.763 |
| Synthetic lifecycle KS | 0.405 |
| Top-decile recall | 34.48% |
| Approval policy acceptance | 76.27% |
| Default rate among approved | 4.20% |
| Default rate among declined | 18.50% |
| Real OpenML benchmark ROC-AUC | 0.801 |
| Real OpenML benchmark KS | 0.494 |

## Why ROC-AUC, PR-AUC, and KS

ROC-AUC measures rank-ordering across all thresholds. PR-AUC is more informative under default-class imbalance because it focuses on precision and recall for the bad/default class. KS is common in credit-risk scorecards because it measures separation between good and bad distributions.

## Why Temporal Validation

Credit behavior drifts over time through acquisition mix, macro stress, utilization, and repayment behavior. A random split can leak future distribution information into training. Splitting by `vintage_month` is closer to how an underwriting model would be evaluated before production.

## How The Approval Policy Is Chosen

The project generates a probability-of-default threshold frontier. Each threshold reports approval rate, approved default rate, declined default rate, average risk-adjusted margin, and total risk-adjusted margin. The selected threshold maximizes total risk-adjusted margin while respecting default-rate and minimum-approval constraints.

## Reason Codes

The reason-code layer maps high-risk feature deviations into adverse-action-style explanations, such as elevated utilization, weak payment-to-minimum ratio, recent missed payments, high debt-to-income, and low credit score. These are not legal adverse-action notices, but they demonstrate the right model-governance thinking.

## Fairness Caveat

The fairness diagnostics check approval-rate ratios, default rates, churn, average PD, and margin across age band, income band, region, segment, and acquisition channel. This is an analytics/governance screen, not legal approval. A real lender would require protected-class policy, adverse-action compliance, reject inference, bureau governance, and legal review.

## Common Interview Questions

**Why use synthetic data at all?**

The synthetic data lets me model lifecycle fields unavailable in small public datasets: churn, expected margin, acquisition channel, utilization trend, and monthly vintage. I then add a real OpenML benchmark to show the modeling workflow also works on public credit-risk data.

**Why does the scorecard beat tree models here?**

The data-generating process is mostly monotonic and scorecard-like: credit score, utilization, delinquencies, income burden, and missed payments drive risk. Logistic regression can rank-order well under that structure and is also more interpretable.

**Would you deploy this?**

No. I would treat it as a portfolio-grade prototype. Real deployment would require real bureau and servicing data, fairness/adverse-action review, reject inference, macro backtesting, monitoring SLAs, challenger models, and business sign-off.
