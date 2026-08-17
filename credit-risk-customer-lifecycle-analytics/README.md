# Credit Risk Scorecard and Portfolio Analytics

Real-data fintech analytics project built on the official Home Credit Default Risk competition dataset. The pipeline combines applications with bureau histories, previous-credit decisions, and installment behavior to estimate probability of default, calibrate scores, evaluate approval policies, and produce governance evidence.

## Evidence

- 307,511 real loan applications
- 1,716,428 bureau records
- 1,670,214 previous applications
- 13,605,401 installment-payment records
- 60 application and behavioral features
- Three-way stratified train/validation/test design with 46,127 untouched test applications
- Logistic scorecard baseline and LightGBM challenger
- Isotonic calibration fitted only on the validation split
- ROC-AUC, PR-AUC, KS, Brier score, and 10-bin expected calibration error
- Approval/default frontier with an explicitly modeled 45% loss-given-default assumption
- Gender fairness report; gender is excluded from model features

## Results

| Metric | Test result |
|---|---:|
| LightGBM ROC-AUC | 0.7830 |
| LightGBM KS | 0.4221 |
| Isotonic ECE | 0.0024 |
| Policy approval rate | 82.23% |
| Approved realized default rate | 4.66% |

## Run

Kaggle competition rules prohibit redistributing the source archive. Download it after accepting the competition rules, then run:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --archive /path/to/home-credit-default-risk.zip
python -m pytest -q
```

## Outputs

| Artifact | Purpose |
|---|---|
| `outputs/home_credit_model_metrics.csv` | Logistic and LightGBM holdout metrics |
| `outputs/home_credit_approval_frontier.csv` | Approval, realized default, and modeled loss trade-offs |
| `outputs/home_credit_fairness.csv` | Group-level score and approval diagnostics |
| `outputs/home_credit_psi_stability.csv` | Train/test PSI stability baseline |
| `outputs/home_credit_feature_importance.csv` | LightGBM feature importance |
| `outputs/home_credit_provenance.csv` | Dataset and relationship counts |
| `reports/home_credit_real_data_validation.md` | Results and claim boundaries |
| `artifacts/home_credit_summary.json` | Machine-readable evidence summary |

## Claim Boundaries

- All application and behavioral records are real Home Credit competition data.
- Expected loss uses a stated 45% LGD assumption; it is modeled, not observed profit.
- The dataset has no application timestamp, so evaluation is stratified rather than out-of-time.
- This is a research and decision-support project, not a deployed underwriting system.

Legacy synthetic modules remain only for backward-compatible tests and are not used by the primary pipeline or resume claims.
