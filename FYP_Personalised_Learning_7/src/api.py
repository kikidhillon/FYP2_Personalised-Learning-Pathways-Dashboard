"""Prototype API for risk, timeline alignment and pathway recommendations."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.pathway_engine import generate_pathway


MODEL_PATH = Path("models/risk_model.joblib")
DATA_DIR = Path("data/processed")
ML_PATH = DATA_DIR / "ml_semester.csv"
ALIGNMENT_PATH = DATA_DIR / "student_timeline_alignment.csv"

app = FastAPI(title="Personalised Learning Pathway API", version="0.2.0")


def records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "prepared_data_available": ML_PATH.exists() and ALIGNMENT_PATH.exists(),
        "risk_model_available": MODEL_PATH.exists(),
    }


@app.get("/students/{student_key}/risk")
def student_risk(student_key: str) -> dict:
    if not MODEL_PATH.exists() or not ML_PATH.exists():
        raise HTTPException(status_code=503, detail="Prepare the data and train the risk model first.")
    bundle = joblib.load(MODEL_PATH)
    data = pd.read_csv(ML_PATH)
    rows = data[data["student_key"].eq(student_key)].sort_values("semester_seq")
    if rows.empty:
        raise HTTPException(status_code=404, detail="Student key not found.")

    latest = rows.iloc[[-1]]
    probability = float(bundle["pipeline"].predict_proba(latest[bundle["features"]])[:, 1][0])
    threshold = float(bundle["threshold"])
    return {
        "student_key": student_key,
        "semester_seq": int(latest.iloc[0]["semester_seq"]),
        "risk_probability": round(probability, 4),
        "decision_threshold": round(threshold, 4),
        "predicted_at_risk": bool(probability >= threshold),
    }


@app.get("/students/{student_key}/timeline")
def student_timeline(student_key: str) -> dict:
    if not ALIGNMENT_PATH.exists():
        raise HTTPException(status_code=503, detail="Run the project data preparation first.")
    alignment = pd.read_csv(ALIGNMENT_PATH)
    rows = alignment[alignment["student_key"].eq(student_key)]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Student timeline not found.")
    fields = [
        "semester_seq",
        "module_code",
        "module",
        "grade",
        "default_semester",
        "course_group",
        "timeline_status",
        "curriculum_match_method",
    ]
    return {"student_key": student_key, "courses": records(rows[fields])}


@app.get("/students/{student_key}/pathway")
def student_pathway(
    student_key: str,
    maximum_credits: int | None = Query(default=None, ge=1, le=24),
) -> dict:
    try:
        result = generate_pathway(student_key, maximum_credits, DATA_DIR)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "summary": result["summary"],
        "recommendations": records(result["recommendations"]),
        "selection_requirements": records(result["selection_requirements"]),
        "held_for_manual_review": records(result["held_for_manual_review"]),
        "deferred_due_requirements": records(result["deferred_due_requirements"]),
        "unmapped_failed_courses": records(result["unmapped_failed_courses"]),
    }
