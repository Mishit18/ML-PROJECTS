# Architecture

```mermaid
flowchart LR
    N0["Home Credit tables"] --> N1["Point-in-time aggregation"]
    N1["Point-in-time aggregation"] --> N2["Scorecard + LightGBM"]
    N2["Scorecard + LightGBM"] --> N3["Calibration + fairness"]
    N3["Calibration + fairness"] --> N4["Approval frontier"]
    N4["Approval frontier"]
```

## Claim boundary

Real public data; approval/loss economics are explicitly modeled assumptions.
