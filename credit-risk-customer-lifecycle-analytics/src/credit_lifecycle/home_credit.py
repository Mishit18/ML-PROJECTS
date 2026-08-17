from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .modeling import ks_statistic
from .monitoring import drift_report
from .real_data import expected_calibration_error


APPLICATION_NUMERIC = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
]
APPLICATION_CATEGORICAL = [
    "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
]
TARGET = "TARGET"


def _read_csv(archive: ZipFile, filename: str, **kwargs) -> pd.DataFrame:
    with archive.open(filename) as stream:
        return pd.read_csv(stream, **kwargs)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _bureau_features(archive: ZipFile) -> pd.DataFrame:
    columns = [
        "SK_ID_CURR",
        "CREDIT_ACTIVE",
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE",
    ]
    bureau = _read_csv(archive, "bureau.csv", usecols=columns)
    bureau["bureau_active"] = bureau["CREDIT_ACTIVE"].eq("Active").astype(int)
    bureau["bureau_overdue"] = bureau["CREDIT_DAY_OVERDUE"].gt(0).astype(int)
    aggregated = bureau.groupby("SK_ID_CURR").agg(
        bureau_credits=("DAYS_CREDIT", "size"),
        bureau_active_credits=("bureau_active", "sum"),
        bureau_overdue_credits=("bureau_overdue", "sum"),
        bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
        bureau_days_credit_min=("DAYS_CREDIT", "min"),
        bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
        bureau_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
    ).reset_index()
    aggregated.attrs["source_rows"] = len(bureau)
    return aggregated


def _previous_application_features(archive: ZipFile) -> pd.DataFrame:
    columns = [
        "SK_ID_CURR",
        "NAME_CONTRACT_STATUS",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "CNT_PAYMENT",
        "DAYS_DECISION",
    ]
    previous = _read_csv(archive, "previous_application.csv", usecols=columns)
    previous["previous_approved"] = previous["NAME_CONTRACT_STATUS"].eq("Approved").astype(int)
    previous["previous_refused"] = previous["NAME_CONTRACT_STATUS"].eq("Refused").astype(int)
    previous["previous_credit_gap"] = previous["AMT_APPLICATION"] - previous["AMT_CREDIT"]
    aggregated = previous.groupby("SK_ID_CURR").agg(
        previous_applications=("DAYS_DECISION", "size"),
        previous_approval_rate=("previous_approved", "mean"),
        previous_refusal_rate=("previous_refused", "mean"),
        previous_days_decision_mean=("DAYS_DECISION", "mean"),
        previous_credit_mean=("AMT_CREDIT", "mean"),
        previous_annuity_mean=("AMT_ANNUITY", "mean"),
        previous_payment_count_mean=("CNT_PAYMENT", "mean"),
        previous_credit_gap_mean=("previous_credit_gap", "mean"),
    ).reset_index()
    aggregated.attrs["source_rows"] = len(previous)
    return aggregated


def _installment_features(archive: ZipFile, chunk_size: int = 1_000_000) -> pd.DataFrame:
    columns = ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    partials: list[pd.DataFrame] = []
    source_rows = 0
    with archive.open("installments_payments.csv") as stream:
        for chunk in pd.read_csv(stream, usecols=columns, chunksize=chunk_size):
            source_rows += len(chunk)
            chunk["days_late"] = (chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]).clip(lower=0)
            chunk["late"] = chunk["days_late"].gt(0).astype(int)
            chunk["underpaid"] = chunk["AMT_PAYMENT"].lt(chunk["AMT_INSTALMENT"] * 0.99).astype(int)
            chunk["payment_ratio"] = _safe_divide(chunk["AMT_PAYMENT"], chunk["AMT_INSTALMENT"]).clip(0, 5)
            grouped = chunk.groupby("SK_ID_CURR").agg(
                installment_count=("late", "size"),
                installment_late_count=("late", "sum"),
                installment_underpaid_count=("underpaid", "sum"),
                installment_days_late_sum=("days_late", "sum"),
                installment_payment_ratio_sum=("payment_ratio", "sum"),
                installment_payment_ratio_count=("payment_ratio", "count"),
            )
            partials.append(grouped)
    combined = pd.concat(partials).groupby(level=0).sum()
    combined["installment_late_rate"] = _safe_divide(
        combined["installment_late_count"], combined["installment_count"]
    )
    combined["installment_underpaid_rate"] = _safe_divide(
        combined["installment_underpaid_count"], combined["installment_count"]
    )
    combined["installment_days_late_mean"] = _safe_divide(
        combined["installment_days_late_sum"], combined["installment_count"]
    )
    combined["installment_payment_ratio_mean"] = _safe_divide(
        combined["installment_payment_ratio_sum"], combined["installment_payment_ratio_count"]
    )
    result = combined.reset_index().drop(
        columns=["installment_days_late_sum", "installment_payment_ratio_sum", "installment_payment_ratio_count"]
    )
    result.attrs["source_rows"] = source_rows
    return result


def build_home_credit_features(archive_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not archive_path.exists():
        raise FileNotFoundError(f"Home Credit archive not found: {archive_path}")
    with ZipFile(archive_path) as archive:
        application_columns = ["SK_ID_CURR", TARGET, "CODE_GENDER", *APPLICATION_NUMERIC, *APPLICATION_CATEGORICAL]
        applications = _read_csv(archive, "application_train.csv", usecols=application_columns)
        bureau = _bureau_features(archive)
        previous = _previous_application_features(archive)
        installments = _installment_features(archive)

    applications["age_years"] = -applications["DAYS_BIRTH"] / 365.25
    applications["employment_years"] = (-applications["DAYS_EMPLOYED"] / 365.25).where(
        applications["DAYS_EMPLOYED"] > -365243
    )
    applications["credit_income_ratio"] = _safe_divide(applications["AMT_CREDIT"], applications["AMT_INCOME_TOTAL"])
    applications["annuity_income_ratio"] = _safe_divide(applications["AMT_ANNUITY"], applications["AMT_INCOME_TOTAL"])
    applications["credit_term_proxy"] = _safe_divide(applications["AMT_CREDIT"], applications["AMT_ANNUITY"])
    applications["external_score_mean"] = applications[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

    frame = applications.merge(bureau, on="SK_ID_CURR", how="left")
    frame = frame.merge(previous, on="SK_ID_CURR", how="left")
    frame = frame.merge(installments, on="SK_ID_CURR", how="left")
    frame["bureau_debt_ratio"] = _safe_divide(frame["bureau_debt_sum"], frame["bureau_credit_sum"])
    for column in APPLICATION_CATEGORICAL:
        frame[column] = frame[column].astype("category")
    provenance = {
        "applications": int(len(applications)),
        "bureau_records": int(bureau.attrs["source_rows"]),
        "previous_application_records": int(previous.attrs["source_rows"]),
        "installment_records": int(installments.attrs["source_rows"]),
        "bureau_customers": int(bureau["SK_ID_CURR"].nunique()),
        "previous_application_customers": int(previous["SK_ID_CURR"].nunique()),
        "installment_customers": int(installments["SK_ID_CURR"].nunique()),
    }
    return frame, provenance


def _split_features(frame: pd.DataFrame):
    train, holdout = train_test_split(frame, test_size=0.30, stratify=frame[TARGET], random_state=42)
    validation, test = train_test_split(holdout, test_size=0.50, stratify=holdout[TARGET], random_state=43)
    excluded = {"SK_ID_CURR", TARGET, "CODE_GENDER"}
    feature_columns = [column for column in frame.columns if column not in excluded]
    categorical = [
        column for column in feature_columns
        if frame[column].dtype == "object" or isinstance(frame[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in feature_columns if column not in categorical]
    return train, validation, test, feature_columns, numeric, categorical


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])


def _metric_row(name: str, y_true: pd.Series, probability: np.ndarray) -> dict[str, float | str]:
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "ks": float(ks_statistic(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "ece_10bin": float(expected_calibration_error(y_true, probability)),
    }


def _approval_frontier(scored: pd.DataFrame, lgd: float = 0.45) -> pd.DataFrame:
    rows = []
    connection = duckdb.connect()
    connection.register("scored", scored)
    for approval_target in np.arange(0.50, 0.96, 0.05):
        threshold = float(
            connection.execute("select quantile_cont(pd_score, ?) from scored", [approval_target]).fetchone()[0]
        )
        result = connection.execute(
            """
            select
                count(*) filter (where pd_score <= ?) as approved_applications,
                avg(case when pd_score <= ? then 1.0 else 0.0 end) as realized_approval_rate,
                avg(TARGET) filter (where pd_score <= ?) as approved_realized_default_rate,
                avg(TARGET) filter (where pd_score > ?) as declined_realized_default_rate,
                sum(pd_score * AMT_CREDIT * ?) filter (where pd_score <= ?) as modeled_expected_loss_lgd45,
                sum(AMT_CREDIT) filter (where pd_score > ?) as modeled_declined_credit_opportunity
            from scored
            """,
            [threshold, threshold, threshold, threshold, lgd, threshold, threshold],
        ).fetchone()
        rows.append({
            "target_approval_rate": approval_target,
            "pd_threshold": threshold,
            "approved_applications": int(result[0]),
            "realized_approval_rate": float(result[1]),
            "approved_realized_default_rate": float(result[2]),
            "declined_realized_default_rate": float(result[3]),
            "modeled_expected_loss_lgd45": float(result[4]),
            "modeled_declined_credit_opportunity": float(result[5]),
        })
    connection.close()
    return pd.DataFrame(rows)


def _fairness_report(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gender, group in scored[scored["CODE_GENDER"].isin(["F", "M"])].groupby("CODE_GENDER"):
        rows.append(
            {
                "gender": gender,
                "applications": int(len(group)),
                "default_rate": float(group[TARGET].mean()),
                "mean_pd": float(group["pd_score"].mean()),
                "approval_rate_at_policy": float(group["approve_policy"].mean()),
            }
        )
    report = pd.DataFrame(rows)
    report["approval_rate_ratio_vs_max"] = report["approval_rate_at_policy"] / report["approval_rate_at_policy"].max()
    return report


def run_home_credit_pipeline(archive_path: Path, output_dir: Path, artifact_dir: Path, report_dir: Path) -> dict[str, float]:
    frame, provenance = build_home_credit_features(archive_path)
    train, validation, test, feature_columns, numeric, categorical = _split_features(frame)

    logistic = Pipeline(
        [("prep", _preprocessor(numeric, categorical)), ("model", LogisticRegression(max_iter=1200, class_weight="balanced"))]
    )
    lightgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    logistic.fit(train[feature_columns], train[TARGET])
    lightgbm.fit(
        train[feature_columns],
        train[TARGET],
        categorical_feature=categorical,
    )
    validation_probability = lightgbm.predict_proba(validation[feature_columns])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(validation_probability, validation[TARGET])

    logistic_probability = logistic.predict_proba(test[feature_columns])[:, 1]
    raw_probability = lightgbm.predict_proba(test[feature_columns])[:, 1]
    calibrated_probability = calibrator.predict(raw_probability)
    metrics = pd.DataFrame(
        [
            _metric_row("logistic_baseline", test[TARGET], logistic_probability),
            _metric_row("lightgbm_raw", test[TARGET], raw_probability),
            _metric_row("lightgbm_isotonic", test[TARGET], calibrated_probability),
        ]
    )

    scored = test[["SK_ID_CURR", TARGET, "CODE_GENDER", "AMT_CREDIT", "AMT_INCOME_TOTAL"]].copy()
    scored["pd_score"] = calibrated_probability
    policy_threshold = float(scored["pd_score"].quantile(0.80))
    scored["approve_policy"] = scored["pd_score"].le(policy_threshold).astype(int)
    frontier = _approval_frontier(scored)
    fairness = _fairness_report(scored)
    stability = drift_report(train, test, numeric)

    importance = pd.DataFrame(
        {"feature": feature_columns, "gain_importance": lightgbm.feature_importances_}
    ).sort_values("gain_importance", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "home_credit_model_metrics.csv", index=False)
    frontier.to_csv(output_dir / "home_credit_approval_frontier.csv", index=False)
    fairness.to_csv(output_dir / "home_credit_fairness.csv", index=False)
    stability.to_csv(output_dir / "home_credit_psi_stability.csv", index=False)
    importance.to_csv(output_dir / "home_credit_feature_importance.csv", index=False)
    pd.DataFrame([provenance]).to_csv(output_dir / "home_credit_provenance.csv", index=False)
    joblib.dump(lightgbm, artifact_dir / "home_credit_lightgbm.joblib")
    joblib.dump(calibrator, artifact_dir / "home_credit_isotonic_calibrator.joblib")

    best = metrics.loc[metrics["model"] == "lightgbm_isotonic"].iloc[0]
    summary = {
        **provenance,
        "test_applications": int(len(test)),
        "features": int(len(feature_columns)),
        "roc_auc": float(best["roc_auc"]),
        "pr_auc": float(best["pr_auc"]),
        "ks": float(best["ks"]),
        "brier_score": float(best["brier_score"]),
        "ece_10bin": float(best["ece_10bin"]),
        "policy_approval_rate": float(scored["approve_policy"].mean()),
        "policy_default_rate": float(scored.loc[scored["approve_policy"] == 1, TARGET].mean()),
    }
    (report_dir / "home_credit_real_data_validation.md").write_text(
        "# Home Credit Real-Data Validation\n\n"
        f"- Applications: **{summary['applications']:,}**\n"
        f"- Holdout applications: **{summary['test_applications']:,}**\n"
        f"- Engineered model features: **{summary['features']}**\n"
        f"- Isotonic LightGBM ROC-AUC: **{summary['roc_auc']:.4f}**\n"
        f"- KS: **{summary['ks']:.4f}**\n"
        f"- ECE (10 bins): **{summary['ece_10bin']:.4f}**\n"
        f"- 80% policy approval rate: **{summary['policy_approval_rate']:.2%}**\n"
        f"- Approved realized default rate: **{summary['policy_default_rate']:.2%}**\n\n"
        "## Claim Boundary\n\n"
        "Application, bureau, previous-credit, and installment records are real competition data. "
        "Expected loss uses an explicit 45% LGD assumption and is therefore modeled, not observed profit. "
        "The source provides no application timestamp, so evaluation uses a stratified three-way split, not out-of-time validation.\n",
        encoding="utf-8",
    )
    (artifact_dir / "home_credit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
