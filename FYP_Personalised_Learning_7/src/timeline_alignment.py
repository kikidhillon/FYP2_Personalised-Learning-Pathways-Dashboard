"""Align each recorded student course with the applicable default programme timeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_access import COURSE_CODE_EQUIVALENTS, normalise_code, normalise_course_name


def _lookup_catalog(catalog: pd.DataFrame) -> tuple[dict, dict]:
    usable = catalog[~catalog["is_placeholder"].fillna(False)].copy()
    usable["timeline_priority"] = usable["source_type"].ne("timeline").astype(int)
    usable = usable.sort_values(["programme_code", "timeline_priority", "curriculum_row_id"])

    code_lookup = {}
    name_lookup = {}
    for row in usable.to_dict("records"):
        code_lookup.setdefault((row["programme_code"], row["course_code"]), row)
        normalised_name = row.get("course_name_normalised", "")
        if normalised_name:
            name_lookup.setdefault((row["programme_code"], normalised_name), row)
    return code_lookup, name_lookup


def _align_rows(rows: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    code_lookup, name_lookup = _lookup_catalog(catalog)
    aligned = rows.copy()
    aligned["module_code_normalised"] = aligned["module_code"].map(normalise_code)
    aligned["module_name_normalised"] = aligned["module"].map(normalise_course_name)

    matches = []
    for row in aligned.to_dict("records"):
        programme = row.get("programme_code")
        code_key = (programme, row["module_code_normalised"])
        name_key = (programme, row["module_name_normalised"])
        match = code_lookup.get(code_key)
        method = "course_code"
        if match is None:
            equivalent_code = COURSE_CODE_EQUIVALENTS.get(row["module_code_normalised"])
            if equivalent_code:
                match = code_lookup.get((programme, equivalent_code))
                method = "equivalent_course_code" if match is not None else "course_code"
        if match is None:
            match = name_lookup.get(name_key)
            method = "course_name" if match is not None else "unmapped"

        matches.append(
            {
                "curriculum_row_id": match.get("curriculum_row_id") if match else pd.NA,
                "curriculum_course_code_raw": match.get("course_code_raw") if match else pd.NA,
                "curriculum_course_name": match.get("course_name") if match else pd.NA,
                "default_semester": match.get("default_semester") if match else pd.NA,
                "course_group": match.get("course_group") if match else pd.NA,
                "curriculum_source_type": match.get("source_type") if match else pd.NA,
                "prerequisite_text": match.get("prerequisite_text") if match else pd.NA,
                "curriculum_match_method": method,
            }
        )
    return pd.concat([aligned.reset_index(drop=True), pd.DataFrame(matches)], axis=1)


def _assign_flexible_slots(alignment: pd.DataFrame, timeline: pd.DataFrame,
                           history: pd.DataFrame | None = None) -> pd.DataFrame:
    result = alignment.copy()
    slot_map: dict[tuple[str, str], list[int]] = {}
    for (programme, group), frame in timeline[
        timeline["course_group"].isin(["EM", "CSp"])
    ].groupby(["programme_code", "course_group"]):
        slot_map[(programme, group)] = list(zip(
            frame["curriculum_row_id"], frame["default_semester"].astype(int)
        ))

    assignments: dict[tuple[str, str], int] = {}
    flexible = result[
        result["curriculum_source_type"].isin(["minor_elective", "core_specialisation"])
    ].sort_values(["student_key", "semester_seq", "module_code_normalised"])

    for (student, programme, group), frame in flexible.groupby(
        ["student_key", "programme_code", "course_group"], dropna=False
    ):
        codes = frame.drop_duplicates("module_code_normalised")["module_code_normalised"].tolist()
        if history is not None and not history.empty:
            previous = history[
                history["student_key"].eq(student)
                & history["programme_code"].eq(programme)
                & history["course_group"].eq(group)
                & history["curriculum_source_type"].isin(["minor_elective", "core_specialisation"])
            ].sort_values("semester_seq")
            old_codes = previous["module_code_normalised"].dropna().drop_duplicates().tolist()
            codes = list(dict.fromkeys(old_codes + codes))
        slots = slot_map.get((programme, group), [])
        for index, code in enumerate(codes):
            if index < len(slots):
                assignments[(student, code)] = slots[index]

    flexible_mask = result["default_semester"].isna() & result["curriculum_source_type"].isin(
        ["minor_elective", "core_specialisation"]
    )
    for index, row in result[flexible_mask].iterrows():
        slot = assignments.get((row["student_key"], row["module_code_normalised"]))
        if slot is not None:
            result.at[index, "curriculum_row_id"] = slot[0]
            result.at[index, "default_semester"] = slot[1]
    # CoQ and MPU2 courses are choices for specific timeline slots, not extra
    # programme credits. Preserve the chosen course name/code, but assign the
    # generic slot's ID and semester for credit and completion accounting.
    slot_rows = timeline[timeline["refer_table"].isin(["CoQ", "MPU_2"])]
    for (student, programme, reference), frame in result[
        result["curriculum_source_type"].isin(["co_curriculum", "mpu2"])
    ].groupby(["student_key", "programme_code", "curriculum_source_type"]):
        target = "CoQ" if reference == "co_curriculum" else "MPU_2"
        slots = slot_rows[
            slot_rows["programme_code"].eq(programme)
            & slot_rows["refer_table"].eq(target)
        ].sort_values("default_semester")
        prior_codes = []
        if history is not None and not history.empty:
            older = history[
                history["student_key"].eq(student)
                & history["curriculum_source_type"].eq(reference)
            ].sort_values("semester_seq")
            prior_codes = older["module_code_normalised"].dropna().drop_duplicates().tolist()
        new_codes = frame.sort_values("semester_seq")["module_code_normalised"].dropna().drop_duplicates().tolist()
        all_codes = list(dict.fromkeys(prior_codes + new_codes))
        for code, slot in zip(all_codes, slots.itertuples()):
            mask = frame["module_code_normalised"].eq(code)
            result.loc[frame.index[mask], "curriculum_row_id"] = slot.curriculum_row_id
            result.loc[frame.index[mask], "default_semester"] = int(slot.default_semester)
    result["default_semester"] = result["default_semester"].astype("Int64")
    return result


def align_course_history(
    courses: pd.DataFrame,
    students: pd.DataFrame,
    catalog: pd.DataFrame,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    student_programmes = students[["student_key", "programme", "programme_code"]].drop_duplicates("student_key")
    rows = courses.merge(student_programmes, on="student_key", how="left", suffixes=("", "_student"))
    rows = _align_rows(rows, catalog)
    rows = _assign_flexible_slots(rows, timeline)
    rows = rows.sort_values(["student_key", "semester_seq", "module_code_normalised"]).reset_index(drop=True)
    rows["attempt_number"] = rows.groupby(["student_key", "module_code_normalised"]).cumcount() + 1
    rows["semester_variance"] = rows["semester_seq"] - rows["default_semester"]

    conditions = [
        rows["attempt_number"].gt(1).fillna(False).to_numpy(dtype=bool),
        rows["default_semester"].isna().to_numpy(dtype=bool),
        rows["semester_variance"].eq(0).fillna(False).to_numpy(dtype=bool),
        rows["semester_variance"].lt(0).fillna(False).to_numpy(dtype=bool),
        rows["semester_variance"].gt(0).fillna(False).to_numpy(dtype=bool),
    ]
    choices = [
        "Repeat attempt",
        "Unmapped to current curriculum",
        "On default semester",
        "Taken earlier",
        "Taken later",
    ]
    rows["timeline_status"] = np.select(conditions, choices, default="Review")
    return rows


def align_current_enrollment(
    enrollment: pd.DataFrame,
    semesters: pd.DataFrame,
    students: pd.DataFrame,
    catalog: pd.DataFrame,
    timeline: pd.DataFrame,
    recorded_alignment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if enrollment.empty:
        return enrollment.copy()
    next_sequence = semesters.groupby("student_key")["semester_seq"].max().add(1)
    rows = enrollment.copy()
    rows["semester_seq"] = rows["student_key"].map(next_sequence).astype("Int64")
    rows = _align_rows(
        rows.merge(
            students[["student_key", "programme", "programme_code"]].drop_duplicates("student_key"),
            on="student_key",
            how="left",
        ),
        catalog,
    )
    rows = _assign_flexible_slots(rows, timeline, recorded_alignment)
    rows["semester_variance"] = rows["semester_seq"] - rows["default_semester"]
    rows["timeline_status"] = np.select(
        [
            rows["default_semester"].isna().to_numpy(dtype=bool),
            rows["semester_variance"].eq(0).fillna(False).to_numpy(dtype=bool),
            rows["semester_variance"].lt(0).fillna(False).to_numpy(dtype=bool),
            rows["semester_variance"].gt(0).fillna(False).to_numpy(dtype=bool),
        ],
        ["Unmapped to current curriculum", "On default semester", "Taken earlier", "Taken later"],
        default="Review",
    )
    return rows


def alignment_summary(
    alignment: pd.DataFrame,
    students: pd.DataFrame,
    semesters: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        alignment.pivot_table(
            index="student_key",
            columns="timeline_status",
            values="module_code",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    counts.columns = [
        "student_key" if column == "student_key" else str(column).lower().replace(" ", "_")
        for column in counts.columns
    ]
    latest = semesters.groupby("student_key", as_index=False)["semester_seq"].max().rename(
        columns={"semester_seq": "latest_recorded_semester"}
    )
    summary = students[["student_key", "programme", "programme_code"]].merge(latest, on="student_key", how="left")
    summary["next_default_semester"] = summary["latest_recorded_semester"].add(1).clip(upper=10).astype("Int64")
    summary = summary.merge(counts, on="student_key", how="left")
    count_columns = [column for column in summary.columns if column not in {"student_key", "programme", "programme_code", "latest_recorded_semester", "next_default_semester"}]
    summary[count_columns] = summary[count_columns].fillna(0).astype(int)
    return summary
