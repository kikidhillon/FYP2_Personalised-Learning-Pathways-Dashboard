"""Prepare all modelling, curriculum and timeline-alignment tables in one command."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.curriculum_loader import load_curriculum_workbook
from src.data_access import CENTRAL_WORKBOOK, CURRICULUM_WORKBOOK, PROCESSED_DIR, load_central_database
from src.timeline_alignment import align_course_history, align_current_enrollment, alignment_summary


def validation_report(
    central: dict[str, pd.DataFrame],
    curriculum: dict[str, pd.DataFrame],
    alignment: pd.DataFrame,
) -> pd.DataFrame:
    students = central["students"]
    ml = central["ml_semester"]
    labelled = ml[ml["next_at_risk_flag"].notna()]
    mapped = alignment["curriculum_match_method"].ne("unmapped")
    timeline = curriculum["curriculum_timeline"]
    totals = timeline.groupby("programme_code")["credit_hours"].sum().to_dict()

    rows = [
        ("students", len(students)),
        ("students_with_sas", int(students["has_sas_file"].fillna(0).sum())),
        ("students_without_sas", int(students["has_sas_file"].eq(0).sum())),
        ("semester_rows", len(ml)),
        ("labelled_semester_rows", len(labelled)),
        ("positive_at_risk_targets", int(labelled["next_at_risk_flag"].sum())),
        ("students_with_positive_target", labelled.loc[labelled["next_at_risk_flag"].eq(1), "student_key"].nunique()),
        ("course_attempt_rows", len(alignment)),
        ("mapped_course_attempt_rows", int(mapped.sum())),
        ("mapped_course_attempt_pct", round(float(mapped.mean() * 100), 2)),
        ("unmapped_course_attempt_rows", int((~mapped).sum())),
        ("it_timeline_credits", totals.get("IT", 0)),
        ("cs_timeline_credits", totals.get("CS", 0)),
        ("is_timeline_credits", totals.get("IS", 0)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def coverage_by_programme(alignment: pd.DataFrame) -> pd.DataFrame:
    frame = alignment.assign(
        mapped=alignment["curriculum_match_method"].ne("unmapped"),
        matched_by_code=alignment["curriculum_match_method"].eq("course_code"),
        matched_by_equivalent_code=alignment["curriculum_match_method"].eq("equivalent_course_code"),
        matched_by_name=alignment["curriculum_match_method"].eq("course_name"),
    )
    result = (
        frame.groupby("programme_code", as_index=False)
        .agg(
            course_attempt_rows=("module_code", "count"),
            mapped_rows=("mapped", "sum"),
            matched_by_code_rows=("matched_by_code", "sum"),
            matched_by_equivalent_code_rows=("matched_by_equivalent_code", "sum"),
            matched_by_name_rows=("matched_by_name", "sum"),
        )
    )
    result["mapped_pct"] = (
        result["mapped_rows"] / result["course_attempt_rows"] * 100
    ).round(2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=CENTRAL_WORKBOOK)
    parser.add_argument("--curriculum", type=Path, default=CURRICULUM_WORKBOOK)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    central = load_central_database(args.database, args.output)
    curriculum = load_curriculum_workbook(args.curriculum, args.output)
    alignment = align_course_history(
        central["sas_courses"],
        central["students"],
        curriculum["curriculum_catalog"],
        curriculum["curriculum_timeline"],
    )
    current_alignment = align_current_enrollment(
        central["current_enrollment"],
        central["sas_semesters"],
        central["students"],
        curriculum["curriculum_catalog"],
        curriculum["curriculum_timeline"],
        alignment,
    )
    summary = alignment_summary(alignment, central["students"], central["sas_semesters"])
    validation = validation_report(central, curriculum, alignment)
    coverage = coverage_by_programme(alignment)
    unmapped_summary = (
        alignment[alignment["curriculum_match_method"].eq("unmapped")]
        .groupby(["programme_code", "module_code", "module"], as_index=False)
        .agg(attempt_rows=("student_key", "count"), students=("student_key", "nunique"))
        .sort_values(["attempt_rows", "students"], ascending=False)
    )

    alignment.to_csv(args.output / "student_timeline_alignment.csv", index=False)
    current_alignment.to_csv(args.output / "current_enrollment_alignment.csv", index=False)
    summary.to_csv(args.output / "student_timeline_summary.csv", index=False)
    validation.to_csv(args.output / "data_validation_summary.csv", index=False)
    coverage.to_csv(args.output / "timeline_alignment_coverage.csv", index=False)
    unmapped_summary.to_csv(args.output / "unmapped_course_summary.csv", index=False)

    print(validation.to_string(index=False))
    print("\nTimeline alignment coverage")
    print(coverage.to_string(index=False))
    print(f"\nPrepared project data in {args.output}")


if __name__ == "__main__":
    main()
