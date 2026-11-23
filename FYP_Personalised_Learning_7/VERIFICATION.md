# Verification Results

The following checks were run against the included workbooks.

## Data preparation

- 97 students loaded.
- Programme mapping: 56 IT, 35 CS and 6 IS.
- 665 semester rows loaded.
- 2,893 historical course-attempt rows loaded and preserved.
- 226 current-enrolment rows loaded and preserved.
- 556 labelled risk rows found, including 24 positive targets from 12 students.
- 2,530 course attempts mapped to the current programme curriculum.
- 363 legacy attempts left unmapped for manual review.
- Mapping coverage: 87.45% overall, 87.35% IT, 87.81% CS and 86.64% IS.
- Curriculum totals: IT 122 credits, CS 123 credits and IS 122 credits.

## Baseline risk model

Evaluation used three-fold `StratifiedGroupKFold` grouped by student.

| Metric | Overall out-of-fold result |
|---|---:|
| Decision threshold | 0.240 |
| PR-AUC | 0.333 |
| Recall | 0.833 |
| Precision | 0.308 |
| F1 | 0.449 |
| Balanced accuracy | 0.874 |
| True positives | 20 |
| False negatives | 4 |
| False positives | 45 |
| True negatives | 487 |

The decision threshold was selected from out-of-fold probabilities to maximise
F1. A nested evaluation would be required for a completely independent
threshold-performance estimate.

## Pathway engine

- Draft pathways were generated successfully for all 96 students who have SAS
  history.
- No generated pathway exceeded its selected maximum credit limit.
- Failed-course repeats receive the highest priority.
- Current enrolment is excluded from duplicate recommendations.
- Minor electives and core-specialisation slots require advisor/student choice.
- Unconfirmed prerequisites and possible historical equivalencies are held for
  manual review.
- S67 is not processed because the central workbook records no SAS history for
  that student.

