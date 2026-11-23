"""Load and normalise the IT, CS and IS semester curriculum workbook."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data_access import CURRICULUM_WORKBOOK, normalise_code, normalise_course_name


PROGRAMMES = ("IT", "CS", "IS")
PLACEHOLDER_CODES = {"MPU2", "MPU4"}


def semester_number(value: object) -> int | None:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def expand_course_codes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [normalise_code(part) for part in str(value).split("/") if part.strip()]


def is_placeholder_code(code: object) -> bool:
    normalised = normalise_code(code)
    return (
        not normalised
        or "X" in normalised
        or normalised in PLACEHOLDER_CODES
        or normalised.startswith("SELECT")
    )


def _timeline_rows(workbook: Path, programme_code: str) -> pd.DataFrame:
    sheet = f"{programme_code}_Timeline"
    frame = pd.read_excel(workbook, sheet_name=sheet, usecols="A:G")
    frame.columns = ["semester", "course_code_raw", "course_name", "credit_hours", "course_group", "prerequisite_text", "refer_table"]
    frame = frame.dropna(subset=["course_code_raw", "course_name"]).copy()
    frame["programme_code"] = programme_code
    frame["default_semester"] = frame["semester"].map(semester_number).astype("Int64")
    frame["source_sheet"] = sheet
    frame["source_type"] = "timeline"
    frame["area"] = pd.NA
    frame["selection_rule"] = pd.NA
    frame["curriculum_row_id"] = [
        f"{programme_code}-TIMELINE-{index + 1:02d}" for index in range(len(frame))
    ]
    frame["is_selection_slot"] = frame["course_code_raw"].map(is_placeholder_code)
    return frame


def _elective_rows(
    workbook: Path,
    programme_code: str,
    source_type: str,
) -> pd.DataFrame:
    suffix = "EM" if source_type == "minor_elective" else "CSp"
    sheet = f"{programme_code}_{suffix}"
    raw = pd.read_excel(workbook, sheet_name=sheet, usecols="A:F")

    if source_type == "minor_elective":
        raw.columns = ["area", "selection_rule", "course_code_raw", "course_name", "credit_hours", "course_group"]
        raw["prerequisite_text"] = pd.NA
    else:
        raw.columns = ["area", "course_code_raw", "course_name", "credit_hours", "course_group", "prerequisite_text"]
        raw["selection_rule"] = pd.NA

    raw = raw.dropna(subset=["course_code_raw", "course_name"]).copy()
    raw["programme_code"] = programme_code
    raw["semester"] = pd.NA
    raw["default_semester"] = pd.Series([pd.NA] * len(raw), dtype="Int64")
    raw["source_sheet"] = sheet
    raw["source_type"] = source_type
    prefix = "EM" if source_type == "minor_elective" else "CSP"
    raw["curriculum_row_id"] = [
        f"{programme_code}-{prefix}-{index + 1:02d}" for index in range(len(raw))
    ]
    raw["is_selection_slot"] = False
    raw["refer_table"] = pd.NA
    return raw


def _referenced_course_rows(workbook: Path, programme_code: str, reference: str) -> pd.DataFrame:
    """Options for MPU2 and CoQ timeline slots, sourced from their referenced tables."""
    sheet = "CoQ" if reference == "CoQ" else "MPU2"
    source_type = "co_curriculum" if reference == "CoQ" else "mpu2"
    group = "CoQ" if reference == "CoQ" else "MPU2"
    raw = pd.read_excel(workbook, sheet_name=sheet, usecols="A:D" if sheet == "CoQ" else "A:E")
    if sheet == "CoQ":
        raw.columns = ["course_code_raw", "course_name", "credit_hours", "prerequisite_text"]
    else:
        raw.columns = ["course_code_raw", "course_name", "credit_hours", "course_group", "prerequisite_text"]
    raw = raw.dropna(subset=["course_code_raw", "course_name"]).copy()
    raw["course_group"] = group
    raw["programme_code"] = programme_code
    raw["semester"] = pd.NA
    raw["default_semester"] = pd.Series([pd.NA] * len(raw), dtype="Int64")
    raw["source_sheet"] = sheet
    raw["source_type"] = source_type
    raw["refer_table"] = reference
    raw["area"] = pd.NA
    raw["selection_rule"] = pd.NA
    raw["curriculum_row_id"] = [
        f"{programme_code}-{group.upper()}-{index + 1:02d}" for index in range(len(raw))
    ]
    raw["is_selection_slot"] = False
    return raw


def _expanded_catalog(rows: pd.DataFrame) -> pd.DataFrame:
    expanded = []
    for row in rows.to_dict("records"):
        for code in expand_course_codes(row["course_code_raw"]):
            item = row.copy()
            item["course_code"] = code
            item["is_placeholder"] = is_placeholder_code(code)
            item["course_name_normalised"] = normalise_course_name(row["course_name"])
            expanded.append(item)
    return pd.DataFrame(expanded)


def load_curriculum_workbook(
    workbook: Path = CURRICULUM_WORKBOOK,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    if not workbook.exists():
        raise FileNotFoundError(f"Curriculum workbook not found: {workbook}")

    required_sheets = {
        f"{programme}_{suffix}"
        for programme in PROGRAMMES
        for suffix in ("Timeline", "EM", "CSp")
    }
    available = set(pd.ExcelFile(workbook).sheet_names)
    missing = required_sheets - available
    if missing:
        raise ValueError(f"Curriculum workbook is missing sheets: {sorted(missing)}")

    timelines = pd.concat(
        [_timeline_rows(workbook, programme) for programme in PROGRAMMES],
        ignore_index=True,
    )
    electives = pd.concat(
        [
            _elective_rows(workbook, programme, source_type)
            for programme in PROGRAMMES
            for source_type in ("minor_elective", "core_specialisation")
        ],
        ignore_index=True,
    )
    referenced_options = pd.concat(
        [
            _referenced_course_rows(workbook, programme, reference)
            for programme in PROGRAMMES
            for reference in ("CoQ", "MPU_2")
        ],
        ignore_index=True,
    )
    electives = pd.concat([electives, referenced_options], ignore_index=True, sort=False)
    all_rows = pd.concat([timelines, electives], ignore_index=True, sort=False)
    catalog = _expanded_catalog(all_rows)
    semester_plan = (
        timelines.groupby(["programme_code", "default_semester"], as_index=False)
        .agg(
            planned_credit_hours=("credit_hours", "sum"),
            planned_slots=("curriculum_row_id", "count"),
        )
        .sort_values(["programme_code", "default_semester"])
    )

    result = {
        "curriculum_timeline": timelines,
        "curriculum_options": electives,
        "curriculum_catalog": catalog,
        "semester_credit_plan": semester_plan,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in result.items():
            frame.to_csv(output_dir / f"{name}.csv", index=False)
    return result

