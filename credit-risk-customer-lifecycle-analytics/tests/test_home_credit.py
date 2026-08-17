import pandas as pd

from credit_lifecycle.home_credit import _approval_frontier, _fairness_report


def test_duckdb_approval_frontier_labels_modeled_economics():
    scored = pd.DataFrame(
        {
            "pd_score": [0.01, 0.04, 0.12, 0.35],
            "TARGET": [0, 0, 1, 1],
            "AMT_CREDIT": [100_000, 120_000, 80_000, 60_000],
        }
    )
    frontier = _approval_frontier(scored)
    assert frontier["realized_approval_rate"].between(0, 1).all()
    assert "modeled_expected_loss_lgd45" in frontier.columns
    assert (frontier["modeled_expected_loss_lgd45"] >= 0).all()


def test_fairness_report_excludes_unknown_gender():
    scored = pd.DataFrame(
        {
            "CODE_GENDER": ["F", "F", "M", "XNA"],
            "TARGET": [0, 1, 0, 1],
            "pd_score": [0.02, 0.20, 0.08, 0.40],
            "approve_policy": [1, 0, 1, 0],
        }
    )
    report = _fairness_report(scored)
    assert set(report["gender"]) == {"F", "M"}
    assert report["approval_rate_ratio_vs_max"].between(0, 1).all()
