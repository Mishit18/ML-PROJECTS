# 2026 Modernization Layer

This service now includes model-governance endpoints in addition to inference and PSI drift:

- `/monitor/retraining_decision` converts drift and latency telemetry into monitor / collect-more-traffic / retrain-candidate actions.
- `/monitor/shadow_agreement` compares champion and challenger probabilities before promotion.
- The design supports champion/challenger deployment language without claiming a full Kubernetes production rollout.
