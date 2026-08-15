# Interview Casebook: Credit Risk and Customer Lifecycle Analytics

## 30-Second Pitch

I built an end-to-end fintech analytics system that scores customer default risk, maps customers into risk bands, simulates approval policy, monitors drift/fairness, and exports SQL-ready business views. The strongest decision artifact is the policy simulator: it shows how approval rate, default risk, expected margin, and declined-customer opportunity cost change as the PD threshold moves.

## Screening-Grade Evidence

| Area | Evidence |
|---|---|
| Model quality | logistic_scorecard ROC-AUC 0.763, KS 0.405, top-decile recall 34.48% |
| Public benchmark | OpenML German Credit benchmark included in `outputs/real_openml_german_credit_benchmark.csv` |
| Policy simulation | `outputs/business_policy_simulator.csv` and `outputs/policy_threshold_frontier.csv` |
| SQL analytics | Risk, cohort, acquisition, retention, pricing, and collections queries in `queries/` |
| Governance | PSI drift, fairness watchlist, reason codes, and model card |

## Business Decision

The best margin-maximizing threshold in this simulation is PD <= 10.00%. It approves 76.27% of customers, expects 120.0 defaults among approved customers, and retains Rs 5,586,754 of risk-adjusted margin.

## Segment Actions

- Safest observed band: A: prime with realized default rate 1.43%.
- Riskiest observed band: E: decline with realized default rate 25.38%.
- Best acquisition pocket: referral / north with 87.23% approval and Rs 2,113 average risk-adjusted margin.
- Highest retention-risk pocket: mass / referral / E: decline with 17.07% churn.
- Fairness watchlist: acquisition_channel=paid_search has approval ratio 0.77 versus the highest-approved peer group.

## SQL Case Questions Covered

1. Which risk bands should we approve, review, or decline?
2. Which acquisition channels produce the best risk-adjusted customers?
3. Which cohorts show deterioration in retention, default, or product depth?
4. Which high-value customers are declined and should be manually reviewed?
5. Which customer pockets need retention campaigns?
6. Which fairness groups require policy investigation?
7. Where does expected margin stop compensating for default risk?

## Interview Defense

If asked whether this is production-ready, the correct answer is no: it is a reproducible analytics and governance prototype. A real lender would require bureau contracts, reject inference, legal review, protected-class testing, challenger monitoring, and live A/B policy evaluation. The project is strong because it shows the full workflow from model score to business decision, not because it pretends synthetic data is a bank's real portfolio.
