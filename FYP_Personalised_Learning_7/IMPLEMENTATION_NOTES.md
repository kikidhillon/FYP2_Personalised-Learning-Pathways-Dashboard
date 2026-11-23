# October 2026 — AI Prompt.docx implementation notes

## Implemented views / placement

| DOCX pages | Screen | Layout implemented |
|---|---|---|
| 1 | Homepage | Campus-photo background; full-viewport UTP header; Academic Dashboard hero above Student / Lecturer entrances; Admin and modal codes |
| 2–5 | Admin cohort | Header/Home, update date, six KPIs, chart pair, performance table, three charts, student roster |
| 8–14 | Admin individual | Three selectors; KPIs; historical/forecast trend; risk bar/credit donut; timeline; semester controls/subjects; failed/incomplete; alignment; next plan |
| 16–20 | Student individual | Student-code restriction; identity text boxes; individual-student layout |
| 21–24 | Lecturer overall | Authorized students only; additional row-select Student Overview above cohort charts |
| 25–29 | Lecturer individual | Restricted name/ID selectors; individual-student layout |

The wireframes are implemented as responsive Streamlit columns. Their section order and group placements follow the DOCX; exact pixel positions will vary with screen width and browser.

## Updated source data

- S1–S97 academic workbook refreshed to the updated student names/IDs.
- Curriculum workbook replaced and processed.
- Student/Lecturer DB added to `data/input`, used for login, display names, student/advisor emails and assignment scope.
- Existing model artefact retained because model feature rows and labels were unchanged (only identifiers changed).

## Data validation

- 97 students; 96 with SAS data, one without.
- 665 semester rows and 2,893 course attempts.
- Last source period: May 2026.
- Programme curriculum totals: IT 122, CS 123, IS 122.
- Academic-advisor assignment: all 97 mapped, with Noureen Talpur (30), Jasmin (28), Azhar (22), and Nurlaili (17). Each lecturer has a distinct six-character alphanumeric code.
- 2,530 mapped attempts (87.45%), 363 unmapped; display notes avoid over-claiming credit completeness.

## Design choices / constraints

1. The DOCX specifies a visual **24%** threshold; model threshold is approximately 23.99%. Both are documented.
2. A Ridge-regression GPA/CGPA forecast was added independently from the existing classification model. No future grades are used to train the per-student forecast; three recorded semesters are required.
3. Historical F/QF course attempts are retained, even if a later attempt passed; Repeat/Incomplete highlights only current unresolved failed attempts.
4. The detailed alignment table retains the fifth real dataset status, `Taken earlier`; the summary bar chart displays the four requested categories and notes the extra status separately.
5. The updated workbook includes some SIT/SIP placements outside default curriculum dates. Historical data are displayed without rewriting, and any incompatible industrial-course co-enrolment is flagged. Future pathway engine now enforces exclusive SIT/SIP terms and consecutiveness based on actual term.
6. Dates for unrecorded future semesters remain blank rather than inventing academic periods/years. Recommended **next** semester uses the recorded term sequence to estimate the next January/May/September date.
7. Source role codes are simple case-sensitive demo codes stored in an Excel workbook. Use Microsoft Entra ID or equivalent institutional SSO before any production deployment involving student academic records.
8. Outlook email notifications were discussed previously but are **not part of the supplied AI Prompt.docx**, and are not added in this change set.
9. The supplied replacement directory includes populated student and advisor emails for all 97 rows. No email delivery action is taken.
10. Course recommendations are prerequisite-checked optimisation with risk context, not a separate trained model that generates a natural-language rationale.

## Verification

- `python -m src.prepare_project_data` succeeded using the updated source workbooks.
- `python -m unittest discover -s tests -v` passed all 7 regression tests.
- Python source tree compiled without syntax errors.
- Streamlit visual launch unavailable in the implementation runtime; install requirements locally and visually inspect desktop/mobile layouts to sign off final placements.

## November 2026 UI and directory refresh

- Replaced `Student_Lecturer_DB.xlsx` with the newly supplied complete lecturer/student mapping; other processed academic records and trained ML model remain unchanged.
- Added `assets/UTP.jpeg`, a photo-backed full-screen home and a full-width university banner followed by the Academic Dashboard heading.
- Updated metric, chart and data table styling; added dedicated Streamlit light/dark palettes in `.streamlit/config.toml`.
- Added distinct suggested-question dropdowns for offline and optional generative chatbot modes; answer choices vary with the active cohort/individual scope, and offline/fallback handling is fact-grounded.
- Added `tests/test_chat_prompts.py`: four checks passed, covering directory/auth, prompt uniqueness, scope isolation and the image asset. All Python sources compiled. Full app rendering and the OR-Tools-dependent suite still need to be checked on a machine with all requirements installed.
