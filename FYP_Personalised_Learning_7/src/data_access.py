"""Shared loaders for the central student database and curriculum workbook."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


CENTRAL_WORKBOOK = Path("data/input/S1_S97_Centralised_Academic_PO_Workbook.xlsx")
CURRICULUM_WORKBOOK = Path("data/input/IT_CS_IS_Semester_Curriculum.xlsx")
PROCESSED_DIR = Path("data/processed")

PROGRAMME_NAME_TO_CODE = {
    "Bachelor of Information Technology (Hons)": "IT",
    "Bachelor of Computer Science (Hons)": "CS",
    "Bachelor of Information Systems (Hons)": "IS",
}
PROGRAMME_CODE_TO_NAME = {value: key for key, value in PROGRAMME_NAME_TO_CODE.items()}

CENTRAL_SHEETS = {
    "Students": "students",
    "ML_Semester": "ml_semester",
    "SAS_Semesters": "sas_semesters",
    "SAS_Courses": "sas_courses",
    "Current_Enrollment": "current_enrollment",
    "SAS_Other_Modules": "sas_other_modules",
    "PO_Achievement": "po_achievement",
    "PO_Courses": "po_courses",
}

FAIL_GRADES = {"F", "QF"}
COURSE_CODE_EQUIVALENTS = {
    "IEB3037": "IFB3037",
    "IFB3037": "IEB3037",
    "IEB3047": "IFB3047",
    "IFB3047": "IEB3047",
    "LEB1042": "LFB1042",
    "LFB1042": "LEB1042",
}


def snake_case(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [snake_case(column) for column in result.columns]
    return result


def normalise_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip().upper())


def normalise_course_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(local|international)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def add_programme_code(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "programme" in result.columns:
        result["programme_code"] = result["programme"].map(PROGRAMME_NAME_TO_CODE)
    return result


def read_central_sheet(
    sheet_name: str,
    workbook: Path = CENTRAL_WORKBOOK,
) -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name=sheet_name)
    return add_programme_code(normalise_columns(frame))


def load_central_database(
    workbook: Path = CENTRAL_WORKBOOK,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    if not workbook.exists():
        raise FileNotFoundError(f"Central workbook not found: {workbook}")

    available = set(pd.ExcelFile(workbook).sheet_names)
    missing = set(CENTRAL_SHEETS) - available
    if missing:
        raise ValueError(f"Central workbook is missing sheets: {sorted(missing)}")

    tables: dict[str, pd.DataFrame] = {}
    for sheet_name, output_name in CENTRAL_SHEETS.items():
        frame = read_central_sheet(sheet_name, workbook)
        tables[output_name] = frame
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_dir / f"{output_name}.csv", index=False)
    return tables


def load_ml_semester(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return read_central_sheet("ML_Semester", path)
    return normalise_columns(pd.read_csv(path))


def is_passing_grade(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().upper() not in FAIL_GRADES
