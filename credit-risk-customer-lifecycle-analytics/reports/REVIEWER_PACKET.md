# External Review Packet

## Claims to verify

- Real benchmark uses 30,000 UCI credit-card customers with a 7,500-record stratified holdout.
- Best uncalibrated GBM reaches ROC-AUC 0.775 and KS 0.421; isotonic calibration lowers 10-bin ECE from 0.01034 to 0.00838.
- Bootstrap ROC-AUC interval for the uncalibrated GBM is 0.762 to 0.789.
- Lifecycle, policy, fairness, PSI, and profit simulations use separately labeled synthetic customers.

## Reproduce

```bash
python scripts/run_pipeline.py
python -m pytest -q
```

## Evidence

- `outputs/real_uci_credit_card_benchmark.csv`
- `outputs/real_uci_calibration_comparison.csv`
- `outputs/real_uci_credit_card_calibration.csv`
- `reports/model_card.md`
- `reports/governance_evidence_pack.md`

## Reviewer checklist

- Confirm target definition, split, metrics, calibration, and bootstrap procedure.
- Confirm no out-of-time claim is made because the dataset has no observation timestamp.
- Review fairness metrics as diagnostics rather than proof of regulatory compliance.
- Confirm synthetic policy economics are labeled modeled or simulated.
- Record reviewer name, role, date, and scope reviewed.
