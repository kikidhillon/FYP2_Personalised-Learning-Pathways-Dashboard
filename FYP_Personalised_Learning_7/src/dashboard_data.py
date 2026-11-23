"""Data helpers used by the Streamlit cohort and individual dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROGRAMME_LABELS = {
    "IT": "Information Technology (IT)",
    "CS": "Computer Science (CS)",
    "IS": "Information Systems (IS)",
}

INDUSTRIAL_COURSE_NAMES = {
    "Student Industrial Training (SIT)": "SIT",
    "Student Industrial Project (SIP)": "SIP",
}


def load_model_bundle(model_path: Path) -> dict[str, Any] | None:
    """Load the trained risk model when it is available."""
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def build_student_snapshot(
    students: pd.DataFrame,
    ml: pd.DataFrame,
    model_bundle: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return one row per student with latest academic and ML-risk fields.

    The student master table is the left side of the merge so students without
    SAS semester history remain visible in the overall cohort dashboard.
    """
    base_columns = [
        "student_key",
        "matric_number",
        "student_name",
        "programme",
        "programme_code",
        "average_po_achievement_pct",
    ]
    base = students[[column for column in base_columns if column in students.columns]].copy()

    latest_columns = [
        "student_key",
        "semester_seq",
        "gpa",
        "cgpa",
        "standing",
        "semester_credit_hours",
        "accumulated_credit_hours",
        "module_count",
        "failed_module_count",
        "current_at_risk_flag",
    ]
    if ml.empty:
        latest_full = pd.DataFrame(columns=latest_columns)
    else:
        latest_full = (
            ml.sort_values(["student_key", "semester_seq"])
            .groupby("student_key", as_index=False)
            .tail(1)
            .copy()
        )
    latest = latest_full[[column for column in latest_columns if column in latest_full.columns]].copy()

    snapshot = base.merge(latest, on="student_key", how="left")
    snapshot["risk_probability"] = pd.NA
    snapshot["predicted_at_risk"] = pd.NA

    if model_bundle is not None and not latest_full.empty:
        features = list(model_bundle.get("features", []))
        usable = latest_full[["student_key", *features]].copy()
        probability = model_bundle["pipeline"].predict_proba(usable[features])[:, 1]
        threshold = float(model_bundle["threshold"])
        predictions = pd.DataFrame(
            {
                "student_key": usable["student_key"].to_numpy(),
                "risk_probability_model": probability,
                "predicted_at_risk_model": probability >= threshold,
            }
        )
        snapshot = snapshot.drop(columns=["risk_probability", "predicted_at_risk"]).merge(
            predictions, on="student_key", how="left"
        )
        snapshot = snapshot.rename(
            columns={
                "risk_probability_model": "risk_probability",
                "predicted_at_risk_model": "predicted_at_risk",
            }
        )

    snapshot["programme_label"] = snapshot["programme_code"].map(PROGRAMME_LABELS)
    return snapshot.sort_values(["programme_code", "student_key"]).reset_index(drop=True)


def programme_performance_summary(
    snapshot: pd.DataFrame,
    alignment: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise academic performance and ML-risk predictions by programme."""
    rows: list[dict[str, Any]] = []
    qf = alignment[
        alignment.get("grade", pd.Series(dtype=str)).astype(str).str.strip().str.upper().eq("QF")
    ].copy()

    for programme_code in ("IT", "CS", "IS"):
        cohort = snapshot[snapshot["programme_code"].eq(programme_code)]
        programme_qf = qf[qf["programme_code"].eq(programme_code)] if not qf.empty else qf
        predicted = cohort[cohort["risk_probability"].notna()]
        at_risk_count = int(predicted["predicted_at_risk"].eq(True).sum())
        coverage = len(predicted)
        rows.append(
            {
                "programme_code": programme_code,
                "Course / Programme": PROGRAMME_LABELS[programme_code],
                "Students": int(cohort["student_key"].nunique()),
                "Avg latest GPA": cohort["gpa"].mean(),
                "Avg latest CGPA": cohort["cgpa"].mean(),
                "Avg failed modules (latest sem)": cohort["failed_module_count"].mean(),
                "Students with QF": int(programme_qf["student_key"].nunique()) if not programme_qf.empty else 0,
                "QF attempts": int(len(programme_qf)),
                "ML predictions available": coverage,
                "Predicted at-risk": at_risk_count,
                "Predicted at-risk rate": at_risk_count / coverage if coverage else pd.NA,
                "Avg risk probability": predicted["risk_probability"].mean() if coverage else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def industrial_semester_locks(timeline: pd.DataFrame) -> pd.DataFrame:
    """Identify fixed SIT/SIP semesters directly from the curriculum timeline."""
    if timeline.empty:
        return pd.DataFrame(
            columns=["programme_code", "default_semester", "course_name", "lock_type", "credit_hours"]
        )
    locks = timeline[timeline["course_name"].isin(INDUSTRIAL_COURSE_NAMES)].copy()
    locks["lock_type"] = locks["course_name"].map(INDUSTRIAL_COURSE_NAMES)
    return locks[
        ["programme_code", "default_semester", "course_name", "lock_type", "credit_hours"]
    ].sort_values(["programme_code", "default_semester"])


def fixed_semester_row(
    timeline: pd.DataFrame,
    programme_code: str,
    semester: int,
) -> pd.DataFrame:
    """Return the SIT/SIP curriculum row when the requested semester is locked."""
    locks = industrial_semester_locks(timeline)
    return locks[
        locks["programme_code"].eq(programme_code)
        & locks["default_semester"].eq(int(semester))
    ].copy()
