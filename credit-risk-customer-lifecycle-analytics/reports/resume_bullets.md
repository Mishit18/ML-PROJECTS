# Resume Bullets

## Data Science / Analytics

- Built end-to-end credit risk and customer lifecycle analytics pipeline on 3,438+ synthetic banking customers, combining bureau, repayment, utilization, acquisition, churn, and margin signals.
- Trained and compared scorecard-style logistic regression, gradient boosting, and random forest models; selected logistic_scorecard with ROC-AUC 0.763, PR-AUC 0.256, KS 0.405, and 34.5% recall in the riskiest decile.
- Designed SQL analytics layer in DuckDB for risk-band policy, approval rate, realized default, churn, segment profitability, and acquisition-channel lifecycle KPIs.
- Added permutation explainability, reason-code generation, PSI drift monitoring, fairness diagnostics, score-band governance, and risk-adjusted margin simulation to translate model outputs into underwriting and retention actions.

## Interview Defense

- The dataset is synthetic for reproducibility, but relationships are intentionally modeled after unsecured credit portfolios.
- The split is temporal by vintage month, avoiding random-split leakage.
- Metrics include ROC-AUC, PR-AUC, KS, top-decile recall, approval policy, and realized default by risk band.
- The project avoids claiming production profit uplift; it reports a policy simulation, threshold frontier, reason codes, fairness diagnostics, real public-data benchmark, and documented governance checks.
