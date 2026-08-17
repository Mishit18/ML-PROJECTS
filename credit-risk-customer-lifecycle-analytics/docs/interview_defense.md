# Interview Defense

## What is real?

All 307,511 applications and 17.0M bureau, previous-loan, and installment records come from the official Home Credit competition archive.

## What is modeled?

Expected loss uses predicted PD, observed credit amount, and an explicit 45% LGD assumption. It is not observed lender profit.

## How was leakage controlled?

Calibration is fitted only on the validation partition. Final metrics are reported on 46,127 untouched applications. Gender is retained only for fairness analysis and excluded from model features.

## Why not temporal validation?

The competition source does not expose an application timestamp. A stratified three-way split is accurate wording; claiming out-of-time validation would be false.

## What would production require?

Reject inference, lender-specific policy constraints, legal review, protected-class governance, live drift monitoring, challenger approval, and audited financial impact.
