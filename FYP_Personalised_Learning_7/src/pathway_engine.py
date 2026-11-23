"""Generate a curriculum-aligned draft pathway for advisor review."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.industrial_rules import planned_industrial_course
from src.portal_data import chronological_enrollment, credit_status, REQUIRED_CREDITS, next_period

from src.data_access import (
    COURSE_CODE_EQUIVALENTS,
    FAIL_GRADES,
    PROCESSED_DIR,
    normalise_code,
    normalise_course_name,
)

try:
    from ortools.sat.python import cp_model
except ImportError:  # The deterministic fallback keeps data preparation usable.
    cp_model = None


def _read(name: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.prepare_project_data first.")
    return pd.read_csv(path)


def _passed(value: object) -> bool:
    return pd.notna(value) and str(value).strip().upper() not in FAIL_GRADES


def _prerequisite_status(
    text: object,
    completed_codes: set[str],
    completed_names: set[str],
    latest_student: pd.Series,
) -> tuple[bool, str]:
    if pd.isna(text) or not str(text).strip():
        return True, "No prerequisite listed"
    requirement = str(text).strip()
    lowered = requirement.lower()
    if "varies by selected" in lowered:
        return False, "Prerequisite depends on the selected specialisation course"
    if "3rd year standing" in lowered:
        completed_semesters = int(latest_student.get("semester_seq", 0) or 0)
        return completed_semesters >= 5, (
            "Third-year standing based on recorded academic semesters"
            if completed_semesters >= 5 else "Third-year standing not yet confirmed"
        )

    if "minimum cgpa" in lowered and "65 credit hours" in lowered:
        cgpa = float(latest_student.get("cgpa", 0) or 0)
        credits = float(latest_student.get("accumulated_credit_hours", 0) or 0)
        semesters = int(latest_student.get("semester_seq", 0) or 0)
        eligible = cgpa >= 2.0 and credits >= 65 and semesters >= 5
        return eligible, (
            f"SIT rule: CGPA {cgpa:.2f}, {credits:.0f} credits and {semesters} completed semesters"
        )

    normalised_requirement_code = normalise_code(requirement)
    equivalent = COURSE_CODE_EQUIVALENTS.get(normalised_requirement_code)
    if normalised_requirement_code in completed_codes or equivalent in completed_codes:
        return True, f"Completed prerequisite {requirement}"

    requirement_name = normalise_course_name(requirement)
    if requirement_name in completed_names:
        return True, f"Completed prerequisite {requirement}"

    for completed_name in completed_names:
        if completed_name and (completed_name in requirement_name or requirement_name in completed_name):
            return True, f"Completed prerequisite {requirement}"
    return False, f"Prerequisite not confirmed: {requirement}"


def _optimise(candidates: pd.DataFrame, available_credits: int, *, use_ortools: bool = True) -> list[str]:
    eligible = candidates[candidates["eligible"]].copy()
    if eligible.empty or available_credits <= 0:
        return []

    if cp_model is None or not use_ortools:
        selected = []
        used = 0
        for row in eligible.sort_values(
            ["priority_score", "credit_hours"], ascending=[False, True]
        ).itertuples():
            credits = int(row.credit_hours)
            if used + credits <= available_credits:
                selected.append(row.curriculum_row_id)
                used += credits
        return selected

    model = cp_model.CpModel()
    take = {
        row.curriculum_row_id: model.NewBoolVar(f"take_{row.curriculum_row_id}")
        for row in eligible.itertuples()
    }
    model.Add(
        sum(
            int(row.credit_hours) * take[row.curriculum_row_id]
            for row in eligible.itertuples()
        )
        <= available_credits
    )
    model.Maximize(
        sum(
            int(row.priority_score) * take[row.curriculum_row_id]
            for row in eligible.itertuples()
        )
    )
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []
    return [row_id for row_id, variable in take.items() if solver.Value(variable) == 1]


def generate_pathway(
    student_key: str,
    maximum_credits: int | None = None,
    data_dir: Path = PROCESSED_DIR,
    *, planning_history: pd.DataFrame | None = None,
    planning_current: pd.DataFrame | None = None,
    planning_semester: int | None = None,
    use_ortools: bool = True,
) -> dict[str, Any]:
    students = _read("students", data_dir)
    ml = _read("ml_semester", data_dir)
    alignment = _read("student_timeline_alignment", data_dir)
    current_alignment = chronological_enrollment(ml, _read("current_enrollment_alignment", data_dir))
    timeline = _read("curriculum_timeline", data_dir)
    options = _read("curriculum_options", data_dir)
    semester_plan = _read("semester_credit_plan", data_dir)

    student_rows = students[students["student_key"].eq(student_key)]
    if student_rows.empty:
        raise KeyError(f"Unknown student key: {student_key}")
    student = student_rows.iloc[0]
    programme = student["programme_code"]

    ml_student = ml[ml["student_key"].eq(student_key)].sort_values("semester_seq")
    if ml_student.empty:
        raise ValueError(f"No SAS semester history is available for {student_key}")
    latest = ml_student.iloc[-1]
    enrolled_semesters = pd.to_numeric(
        current_alignment.loc[current_alignment["student_key"].eq(student_key), "semester_seq"],
        errors="coerce",
    ).dropna()
    active_semester = int(planning_semester) if planning_semester is not None else max(
        int(latest["semester_seq"]),
        int(enrolled_semesters.max()) if not enrolled_semesters.empty else 0,
    )
    next_semester = active_semester + 1

    planned = semester_plan[
        semester_plan["programme_code"].eq(programme)
        & semester_plan["default_semester"].eq(next_semester)
    ]
    planned_credits = int(planned.iloc[0]["planned_credit_hours"]) if not planned.empty else 15
    requested_maximum_credits = int(maximum_credits) if maximum_credits is not None else None
    maximum_credits = int(maximum_credits or planned_credits)

    history = planning_history.copy() if planning_history is not None else alignment[alignment["student_key"].eq(student_key)].copy()
    history["course_key"] = history["curriculum_row_id"].fillna(history["module_code_normalised"])
    latest_attempts = history.sort_values("semester_seq").drop_duplicates("course_key", keep="last")
    passed = latest_attempts[latest_attempts["grade"].map(_passed)]
    failed = latest_attempts[~latest_attempts["grade"].map(_passed)]
    completed_row_ids = set(passed["curriculum_row_id"].dropna())
    completed_codes = set(passed["module_code_normalised"].dropna())
    completed_names = set(passed["module_name_normalised"].dropna())

    current = planning_current.copy() if planning_current is not None else current_alignment[current_alignment["student_key"].eq(student_key)].copy()
    current_row_ids = set(current.get("curriculum_row_id", pd.Series(dtype=str)).dropna())
    current_codes = set(current.get("module_code_normalised", pd.Series(dtype=str)).dropna())

    programme_timeline = timeline[timeline["programme_code"].eq(programme)].copy()
    fixed_semester_rows = planned_industrial_course(
        timeline, programme, next_semester, history, current
    )
    fixed_semester = not fixed_semester_rows.empty
    fixed_semester_course = (
        " / ".join(fixed_semester_rows["course_name"].astype(str).tolist())
        if fixed_semester
        else None
    )

    if fixed_semester:
        # SIT/SIP are fixed curriculum semesters. The pathway must contain only
        # the industrial course and cannot be filled with overdue/elective work.
        due = fixed_semester_rows.copy()
        maximum_credits = int(fixed_semester_rows["credit_hours"].sum())
    else:
        due = programme_timeline[
            programme_timeline["default_semester"].le(next_semester)
            & ~programme_timeline["course_name"].isin(
                ["Student Industrial Training (SIT)", "Student Industrial Project (SIP)"]
            )
        ].copy()

    passed_group_counts = passed[
        passed["course_group"].isin(["EM", "CSp", "CoQ", "MPU2"])
    ].groupby("course_group")["course_key"].nunique().to_dict()
    current_group_counts = (
        current[current["course_group"].isin(["EM", "CSp", "CoQ", "MPU2"])]
        .groupby("course_group")["module_code_normalised"]
        .nunique()
        .to_dict()
        if not current.empty and "course_group" in current.columns
        else {}
    )
    selection_candidate_rows = []
    for group, label, source_type in [
        ("EM", "Minor Elective", "minor_elective"),
        ("CSp", "Core Specialisation", "core_specialisation"),
        ("CoQ", "Co Curriculum", "co_curriculum"),
        ("MPU2", "MPU2", "mpu2"),
    ]:
        due_slots = due[
            due["refer_table"].eq("CoQ" if group == "CoQ" else "MPU_2")
            if group in ("CoQ", "MPU2") else due["course_group"].eq(group)
        ].sort_values("default_semester")
        completed_count = int(passed_group_counts.get(group, 0)) + int(
            current_group_counts.get(group, 0)
        )
        incomplete_slots = due_slots.iloc[completed_count:]
        available_options = options[
            options["programme_code"].eq(programme)
            & options["source_type"].eq(source_type)
        ]
        available_options = available_options[
            ~available_options["course_code_raw"].map(normalise_code).isin(
                completed_codes | current_codes
            )
        ]
        for slot in incomplete_slots.itertuples():
            overdue = int(slot.default_semester) < next_semester
            selection_candidate_rows.append(
                {
                    "curriculum_row_id": slot.curriculum_row_id,
                    "candidate_kind": "selection",
                    "course_code": f"SELECT_{group.upper()}",
                    "course_name": f"Select {label} course",
                    "credit_hours": int(slot.credit_hours),
                    "course_group": group,
                    "default_semester": int(slot.default_semester),
                    "eligible": len(available_options) > 0,
                    "prerequisite_status": f"{len(available_options)} curriculum options available",
                    "priority_score": 600 if overdue else 400,
                    "reason": (
                        f"Overdue {label} slot from Semester {int(slot.default_semester)}"
                        if overdue
                        else f"{label} slot scheduled in Semester {next_semester}"
                    ),
                    "eligible_option_count": len(available_options),
                    "source_sheet": (
                        f"{programme}_{group}" if group in ("EM", "CSp")
                        else "CoQ" if group == "CoQ" else "MPU2"
                    ),
                }
            )

    failed_ids = set(failed["curriculum_row_id"].dropna())
    unmapped_history_semesters = set(
        history.loc[
            history["curriculum_match_method"].eq("unmapped"), "semester_seq"
        ].dropna().astype(int)
    )
    candidate_rows = []
    for row in due.itertuples():
        if bool(row.is_selection_slot):
            continue
        if row.curriculum_row_id in completed_row_ids or row.curriculum_row_id in current_row_ids:
            continue

        eligible, prerequisite_reason = _prerequisite_status(
            row.prerequisite_text,
            completed_codes,
            completed_names,
            latest,
        )
        if fixed_semester and row.course_name == "Student Industrial Project (SIP)":
            # A booked/current SIT is not a completed prerequisite; reserve
            # next-semester SIP conditionally so it stays the only course.
            from src.industrial_rules import industrial_record
            industrial_taken = industrial_record(history, current)
            if industrial_taken.get("SIT") == next_semester - 1 and not any(
                str(code).upper() in completed_codes for code in ("IFB3037", "IEB3037")
            ):
                eligible = True
                prerequisite_reason = "Conditional: successfully complete current SIT before SIP enrollment"
        if fixed_semester:
            reason = (
                f"Exclusive industrial Semester {next_semester}; "
                "no other courses may be added"
            )
            if "Conditional:" in prerequisite_reason:
                reason += "; subject to successful completion of SIT"
            priority = 2000
        elif row.curriculum_row_id in failed_ids:
            reason = "Repeat failed course"
            priority = 1000
        elif (
            int(row.default_semester) < next_semester
            and int(row.default_semester) in unmapped_history_semesters
        ):
            reason = "Possible legacy-curriculum gap"
            priority = 650
            eligible = planning_history is not None
            prerequisite_reason = (
                "Provisional repeat/unrecorded requirement: advisor must verify whether "
                "an unmapped historical course already satisfies it"
                if planning_history is not None else
                "Advisor must confirm whether an unmapped historical course satisfies this requirement"
            )
        elif int(row.default_semester) < next_semester:
            reason = "Incomplete course from an earlier default semester"
            priority = 700
        else:
            reason = f"Scheduled in default Semester {next_semester}"
            priority = 500

        candidate_rows.append(
            {
                "curriculum_row_id": row.curriculum_row_id,
                "candidate_kind": "named_course",
                "course_code": row.course_code_raw,
                "course_name": row.course_name,
                "credit_hours": int(row.credit_hours),
                "course_group": row.course_group,
                "default_semester": int(row.default_semester),
                "eligible": eligible,
                "prerequisite_status": prerequisite_reason,
                "priority_score": priority,
                "reason": reason,
                "eligible_option_count": pd.NA,
                "source_sheet": row.source_sheet,
            }
        )

    candidates = pd.DataFrame(candidate_rows + selection_candidate_rows)
    # Fixed SIT/SIP semesters must remain exclusive in the provisional pathway.
    # A failed/unverified prerequisite stays flagged as ineligible and held for
    # advisor review; the future projection may model the course conditionally.
    selected_ids = (
        candidates["curriculum_row_id"].tolist() if fixed_semester
        else _optimise(candidates, maximum_credits, use_ortools=use_ortools) if not candidates.empty else []
    )
    selected = (
        candidates[candidates["curriculum_row_id"].isin(selected_ids)].copy()
        if not candidates.empty
        else candidates
    )
    recommendations = (
        selected[selected["candidate_kind"].eq("named_course")]
        .sort_values(["priority_score", "default_semester"], ascending=[False, True])
        .reset_index(drop=True)
        if not selected.empty
        else selected
    )
    selection_frame = (
        selected[selected["candidate_kind"].eq("selection")][
            [
                "course_group",
                "course_name",
                "default_semester",
                "credit_hours",
                "eligible_option_count",
                "source_sheet",
                "reason",
            ]
        ]
        .rename(columns={"course_name": "requirement"})
        .reset_index(drop=True)
        if not selected.empty
        else selected
    )
    held_for_review = (
        candidates[~candidates["eligible"]].reset_index(drop=True)
        if not candidates.empty
        else candidates
    )
    deferred = (
        candidates[
            candidates["eligible"]
            & ~candidates["curriculum_row_id"].isin(selected_ids)
        ].reset_index(drop=True)
        if not candidates.empty
        else candidates
    )
    reserved_credits = (
        int(selection_frame["credit_hours"].sum()) if not selection_frame.empty else 0
    )
    unmapped_failed = failed[failed["curriculum_match_method"].eq("unmapped")][
        ["module_code", "module", "grade", "semester_seq"]
    ].copy()

    risk_flag = latest.get("current_at_risk_flag")
    summary = {
        "student_key": student_key,
        "programme": student["programme"],
        "programme_code": programme,
        "latest_recorded_semester": int(latest["semester_seq"]),
        "next_default_semester": next_semester,
        "latest_cgpa": None if pd.isna(latest.get("cgpa")) else float(latest["cgpa"]),
        "latest_standing": latest.get("standing"),
        "currently_at_risk": None if pd.isna(risk_flag) else bool(risk_flag),
        "requested_maximum_credits": requested_maximum_credits,
        "maximum_credits": maximum_credits,
        "fixed_industrial_semester": fixed_semester,
        "fixed_industrial_course": fixed_semester_course,
        "additional_courses_locked": fixed_semester,
        "reserved_elective_credits": reserved_credits,
        "recommended_named_course_credits": int(recommendations["credit_hours"].sum()) if not recommendations.empty else 0,
        "advisor_review_required": True,
    }
    return {
        "summary": summary,
        "recommendations": recommendations,
        "selection_requirements": selection_frame,
        "held_for_manual_review": held_for_review,
        "deferred_due_requirements": deferred,
        "unmapped_failed_courses": unmapped_failed,
        "planning_selections": selected.copy(),
    }



def generate_future_pathway(student_key: str, maximum_credits: int, data_dir: Path = PROCESSED_DIR,
                            *, initial: dict[str, Any] | None = None,
                            max_future_semesters: int = 25,
                            future_maximum_credits: int | None = None,
                            _use_ortools: bool = True) -> dict[str, Any]:
    """Provisional future terms; stop at the target or once the plan makes no progress.

    The initial, already-displayed next semester retains its own credit limit.
    ``future_maximum_credits`` only controls subsequent semesters. The plan
    never treats unrecorded recommendations as actual earned credits.
    """
    students = _read('students', data_dir)
    programme = str(students.loc[students.student_key.eq(student_key), 'programme_code'].iloc[0])
    timeline = _read('curriculum_timeline', data_dir)
    alignment = _read('student_timeline_alignment', data_dir)
    history = alignment[alignment.student_key.eq(student_key)].copy()
    ml = _read('ml_semester', data_dir)
    latest = ml[ml.student_key.eq(student_key)].sort_values('semester_seq').iloc[-1]
    enrollment = chronological_enrollment(ml, _read('current_enrollment_alignment', data_dir))
    current = enrollment[enrollment.student_key.eq(student_key)].copy()
    active = max(int(latest.semester_seq), int(current.semester_seq.max()) if not current.empty else 0)
    reference = current.sort_values('semester_seq').iloc[-1] if not current.empty else latest
    year, period = next_period(int(reference.year), str(reference.period))
    result = initial if initial is not None else generate_pathway(student_key, maximum_credits, data_dir)
    first_result = result
    projection = []
    incomplete = False
    # Include current enrollments as *hypothetical passes* without changing records.
    # They were excluded by the next-semester engine already.
    simulated_rows = []
    for row in current.itertuples():
        if pd.notna(getattr(row, 'curriculum_row_id', None)):
            simulated = {c: pd.NA for c in history.columns}
            simulated.update(student_key=student_key, semester_seq=int(row.semester_seq),
                curriculum_row_id=row.curriculum_row_id, module_code=row.module_code,
                module_code_normalised=row.module_code_normalised, module=row.module,
                module_name_normalised=row.module_name_normalised, course_group=row.course_group,
                grade='P', attempt_number=99, curriculum_match_method='planned')
            simulated_rows.append(simulated)
    if simulated_rows:
        history = pd.concat([history, pd.DataFrame(simulated_rows)], ignore_index=True)
    start_semester = active + 1
    future_credit_limit = (int(future_maximum_credits) if future_maximum_credits is not None
                           else 16 if result['summary']['fixed_industrial_semester'] else int(maximum_credits))
    if future_credit_limit < 1:
        raise ValueError('The future-semester maximum credit hours must be positive.')
    last_semester = start_semester - 1
    seen_selections = set()
    for offset in range(max_future_semesters + 1):
        previous_credits = credit_status(history, timeline, programme)
        if previous_credits['complete']:
            break
        semester = start_semester + offset
        if offset:
            result = generate_pathway(student_key, future_credit_limit, data_dir,
                planning_history=history, planning_current=current.iloc[0:0], planning_semester=semester-1,
                use_ortools=_use_ortools)
        selected = result['planning_selections']
        if selected.empty:
            incomplete = True
            break
        # A recurring row ID is not a fresh degree requirement. Do not create
        # Semester 14..34 with identical elective placeholders or courses.
        fingerprint = tuple(sorted(str(row_id) for row_id in selected.curriculum_row_id))
        if fingerprint in seen_selections:
            incomplete = True
            break
        seen_selections.add(fingerprint)
        planned = []
        simulated_rows = []
        for row in selected.itertuples():
            planned.append({'Semester Number':semester, 'Semester':period, 'Year':int(year),
                'Course Code':row.course_code, 'Course Name':row.course_name,
                'Credit Hours':int(row.credit_hours), 'Course Group':row.course_group,
                'Eligible':bool(row.eligible), 'Prerequisite':row.prerequisite_status,
                'Priority Score':int(row.priority_score), 'Reason':row.reason})
            simulated = {c:pd.NA for c in history.columns}
            simulated.update(student_key=student_key, semester_seq=semester,
                curriculum_row_id=row.curriculum_row_id, module_code=row.course_code,
                module_code_normalised=normalise_code(row.course_code), module=row.course_name,
                module_name_normalised=normalise_course_name(row.course_name), course_group=row.course_group,
                grade='P', attempt_number=99, curriculum_match_method='planned')
            simulated_rows.append(simulated)
        projected_history = pd.concat([history, pd.DataFrame(simulated_rows)], ignore_index=True)
        projected_credits = credit_status(projected_history, timeline, programme)
        # Even distinct-looking plans cannot be extended indefinitely when
        # they do not increase credits toward the programme requirement.
        if projected_credits['passed'] <= previous_credits['passed']:
            incomplete = True
            break
        history = projected_history
        last_semester = semester
        if offset:
            projection.extend(planned)
        if projected_credits['complete']:
            break
        year, period = next_period(year, period)
    else:
        incomplete = True
    final_credits = credit_status(history, timeline, programme)
    incomplete = incomplete or not final_credits['complete']
    output = {'courses':pd.DataFrame(projection), 'passed_credits_planned':final_credits['passed'],
              'required_credits':REQUIRED_CREDITS[programme], 'incomplete':incomplete,
              'next_semester':start_semester, 'last_semester':last_semester,
              'planning_method': 'OR-Tools' if _use_ortools and cp_model is not None else 'deterministic fallback'}

    # The solver optimises one semester at a time, so a locally high-scoring
    # choice can prevent a complete later pathway. Preserve the displayed
    # first semester and retry only the future terms with deterministic course
    # ordering if the OR-Tools projection ends incomplete. No global state is
    # modified, making the retry safe for concurrent dashboard users.
    if incomplete and _use_ortools and cp_model is not None:
        repaired = generate_future_pathway(
            student_key, maximum_credits, data_dir,
            initial=first_result,
            max_future_semesters=max_future_semesters,
            future_maximum_credits=future_maximum_credits,
            _use_ortools=False,
        )
        if not repaired['incomplete']:
            repaired['planning_method'] = 'deterministic recovery after OR-Tools'
            return repaired
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--max-credits", type=int)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    result = generate_pathway(args.student, args.max_credits, args.data_dir)
    print(pd.Series(result["summary"]).to_string())
    for key in (
        "recommendations",
        "selection_requirements",
        "held_for_manual_review",
        "deferred_due_requirements",
        "unmapped_failed_courses",
    ):
        print(f"\n{key.replace('_', ' ').title()}")
        frame = result[key]
        print("None" if frame.empty else frame.to_string(index=False))


if __name__ == "__main__":
    main()
