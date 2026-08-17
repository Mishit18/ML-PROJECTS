# Model Card

## Intended Use

Rank-order applicants/customers by 12-month default risk and support portfolio-level decisions across approval policy, acquisition channel quality, retention, and risk-adjusted margin.

## Data

Synthetic portfolio with bureau behavior, income, utilization, repayment behavior, card tenure, digital engagement, acquisition channel, region, churn, expected margin, and default labels.

## Metrics

| model              |   roc_auc |   pr_auc |      ks |   f1_top20 |   precision_top20 |   recall_top20 |   recall_top10 |   threshold_top20 |   tp |   fp |   fn |   tn |
|:-------------------|----------:|---------:|--------:|-----------:|------------------:|---------------:|---------------:|------------------:|-----:|-----:|-----:|-----:|
| logistic_scorecard |   0.76329 |  0.25578 | 0.40511 |    0.29294 |           0.20203 |        0.53257 |        0.34483 |           0.11305 |  139 |  549 |  122 | 2628 |
| random_forest      |   0.75305 |  0.22864 | 0.38726 |    0.2803  |           0.19331 |        0.50958 |        0.341   |           0.49187 |  133 |  555 |  128 | 2622 |
| gradient_boosting  |   0.75183 |  0.23035 | 0.37942 |    0.2803  |           0.19331 |        0.50958 |        0.32184 |           0.11033 |  133 |  555 |  128 | 2622 |

## Real Public Dataset Benchmark

| dataset                            | model                   |   records |   test_records |   default_rate |   roc_auc |   pr_auc |      ks |   brier_score |   bad_rate |
|:-----------------------------------|:------------------------|----------:|---------------:|---------------:|----------:|---------:|--------:|--------------:|-----------:|
| UCI Default of Credit Card Clients | gradient_boosting       |     30000 |           7500 |         0.2212 |   0.77483 |  0.55537 | 0.42077 |       0.1356  |      nan   |
| UCI Default of Credit Card Clients | logistic_scorecard      |     30000 |           7500 |         0.2212 |   0.71553 |  0.50231 | 0.37065 |       0.14558 |      nan   |
| OpenML credit-g data_id=31         | real_logistic_scorecard |      1000 |            300 |       nan      |   0.80106 |  0.64467 | 0.49365 |     nan       |        0.3 |
| OpenML credit-g data_id=31         | real_gradient_boosting  |      1000 |            300 |       nan      |   0.76466 |  0.57531 | 0.42857 |     nan       |        0.3 |

## Policy Frontier

|   pd_threshold |   approval_rate |   approved_customers |   declined_customers |   approved_default_rate |   declined_default_rate |   avg_risk_adjusted_margin |   total_risk_adjusted_margin |   avg_expected_margin |
|---------------:|----------------:|---------------------:|---------------------:|------------------------:|------------------------:|---------------------------:|-----------------------------:|----------------------:|
|           0.02 |        0.118965 |                  409 |                 3029 |               0.0122249 |               0.0845163 |                    2714.92 |                  1.1104e+06  |               2895.86 |
|           0.03 |        0.244328 |                  840 |                 2598 |               0.0142857 |               0.095843  |                    2742.34 |                  2.30357e+06 |               3011.56 |
|           0.04 |        0.355439 |                 1222 |                 2216 |               0.0155483 |               0.109206  |                    2686.11 |                  3.28243e+06 |               3035.63 |
|           0.05 |        0.453752 |                 1560 |                 1878 |               0.0275641 |               0.116081  |                    2568.54 |                  4.00692e+06 |               3001.84 |
|           0.06 |        0.544212 |                 1871 |                 1567 |               0.0288616 |               0.1321    |                    2492.58 |                  4.66363e+06 |               3008.58 |
|           0.07 |        0.618092 |                 2125 |                 1313 |               0.0324706 |               0.14623   |                    2410.29 |                  5.12187e+06 |               2998.37 |
|           0.08 |        0.677138 |                 2328 |                 1110 |               0.0347938 |               0.162162  |                    2313.91 |                  5.38679e+06 |               2962.37 |
|           0.09 |        0.722222 |                 2483 |                  955 |               0.0386629 |               0.172775  |                    2217.79 |                  5.50678e+06 |               2921.67 |
|           0.1  |        0.762653 |                 2622 |                  816 |               0.0419527 |               0.185049  |                    2130.72 |                  5.58675e+06 |               2891.56 |
|           0.11 |        0.791449 |                 2721 |                  717 |               0.0448365 |               0.193863  |                    2049.7  |                  5.57723e+06 |               2858.31 |
|           0.12 |        0.819663 |                 2818 |                  620 |               0.0454223 |               0.214516  |                    1968.55 |                  5.54737e+06 |               2826.52 |
|           0.13 |        0.837696 |                 2880 |                  558 |               0.0486111 |               0.216846  |                    1917.77 |                  5.52319e+06 |               2807.15 |

## Limitations

- Synthetic data is useful for reproducibility and interview demonstration, not direct production deployment.
- A real lender would require protected-class fairness review, reject inference, bureau governance, macroeconomic validation, and legal/compliance approval.
- The policy simulation is directional; it does not claim actual business uplift without live experimentation.
