from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


@st.cache_data
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / name)


st.set_page_config(page_title="Credit Risk Lifecycle Analytics", layout="wide")
st.title("Credit Risk and Customer Lifecycle Analytics")

metrics = read_csv("model_metrics.csv")
policy = read_csv("policy_threshold_frontier.csv")
risk = read_csv("risk_band_policy.csv")
channels = read_csv("channel_lifecycle.csv")
segments = read_csv("segment_profitability.csv")
fairness = read_csv("fairness_group_metrics.csv")
reasons = read_csv("adverse_action_reason_codes.csv")
drift = read_csv("drift_report.csv")
real = read_csv("real_openml_german_credit_benchmark.csv")

best = metrics.iloc[0]
cols = st.columns(5)
cols[0].metric("Best Model", best["model"])
cols[1].metric("ROC-AUC", f"{best['roc_auc']:.3f}")
cols[2].metric("KS", f"{best['ks']:.3f}")
cols[3].metric("Top-Decile Recall", f"{best['recall_top10']:.1%}")
cols[4].metric("Public Benchmark AUC", f"{real.iloc[0]['roc_auc']:.3f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Risk Policy", "Lifecycle", "Fairness", "Reason Codes", "Monitoring"])

with tab1:
    st.subheader("Risk Band Policy")
    st.dataframe(risk, use_container_width=True)
    st.line_chart(policy.set_index("approval_rate")[["approved_default_rate", "avg_risk_adjusted_margin"]])

with tab2:
    st.subheader("Acquisition Channel Quality")
    st.dataframe(channels, use_container_width=True)
    st.subheader("Segment Profitability")
    st.dataframe(segments, use_container_width=True)

with tab3:
    st.subheader("Fairness / Compliance Watchlist")
    st.dataframe(fairness.sort_values("approval_rate_ratio_vs_max"), use_container_width=True)

with tab4:
    st.subheader("Adverse-Action Style Reason Codes")
    st.dataframe(reasons, use_container_width=True)

with tab5:
    st.subheader("Feature Drift")
    st.dataframe(drift, use_container_width=True)
    st.subheader("Real Public Dataset Benchmark")
    st.dataframe(real, use_container_width=True)
