# Methodology

## Data Design

The dataset is generated to resemble unsecured lending and card portfolios. Features cover application quality, bureau risk, repayment behavior, utilization stress, product depth, digital engagement, acquisition source, and region.

## Validation

The split is temporal by `vintage_month`: earlier cohorts train the model and later cohorts test it. This is closer to production model validation than a random split because credit portfolios drift over time.

## Modeling

The pipeline compares:

- Logistic regression as an interpretable scorecard baseline
- Gradient boosting for non-linear interactions
- Random forest for robust benchmark comparison

## Metrics

- ROC-AUC: rank-ordering quality
- PR-AUC: performance under default-class imbalance
- KS statistic: scorecard separation
- Recall in riskiest decile: collections / underwriting usefulness
- Top-20 policy precision and recall: approval or manual-review threshold behavior

## Business Layer

The project converts model probability into risk bands and a simulated approval policy. It then connects risk to churn, expected margin, acquisition channel quality, and customer segment profitability.

## Monitoring

Population stability index compares development-window features against the latest scoring month. PSI is reported as stable, moderate, or severe to support model governance.
