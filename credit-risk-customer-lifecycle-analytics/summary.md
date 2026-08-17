# Project Summary

Built a real-data credit-risk and portfolio analytics pipeline on 307,511 Home Credit applications joined to 17.0M bureau, previous-loan, and installment-payment records. The system engineers 60 features, compares logistic scorecard and LightGBM models, calibrates PD estimates, and exports DuckDB approval-policy, fairness, PSI, and governance evidence.

Holdout results: LightGBM ROC-AUC 0.7830, KS 0.4221, and isotonic ECE 0.0024 on 46,127 applications. An 82.2% approval policy yielded 4.66% observed default among approved holdout applications. Expected loss is explicitly modeled under a 45% LGD assumption.
