# UTP Academic Dashboard — Personalised Learning Pathway Advisor

Streamlit research prototype implementing the **29-page `AI Prompt.docx`** specification. It keeps the existing early-risk classification pipeline, course alignment and pathway optimiser, and adds the three role-specific entrances and dashboards.

## Run on Windows (Python 3.10/3.11 recommended)

Extract the ZIP, open a terminal in the project root (the folder containing `app.py`) and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open **http://localhost:8501** if the browser does not open automatically.

The package includes the current `data/processed/` tables and trained risk model. To reprocess the included Excel sources after editing/replacing them:

```powershell
python -m src.prepare_project_data
```

If the **academic values** or training distribution change, retrain the risk model too:

```powershell
python -m src.train_risk_model
```

Check the app again after reprocessing. The last-updated label reflects the latest Year/Period in the prepared academic data (not the computer's current date). Data caches refresh when processed CSV files change.

## Login and roles

The homepage uses the supplied **UTP campus image** as a full-screen background, a **full-viewport UNIVERSITI TEKNOLOGI PETRONAS** header, an **Academic Dashboard** title between the header and Student/Lecturer entrance buttons, and a separate Admin access button. Each entrance opens a modal requesting a code.

| Role | Validation | Authorized views |
|---|---|---|
| Admin | Code from DOCX specification | Overall cohort (97 students) and individual-student dashboards |
| Student | `Student_Code` in `data/input/Student_Lecturer_DB.xlsx` | Only the matching student's individual dashboard |
| Lecturer | `Lecturer_Code` in the same workbook | Only lecturer-assigned students, in advisor overall and advisor individual dashboards |

The specification's demo codes work, including `adminUTP97`, student `k3D4p2` (S1) and lecturer `A1B2C3` (Noureen Talpur). **Access codes are only a prototype feature.** The workbook stores plaintext codes, the admin code is part of the application, and Streamlit role sessions are not institutional security. **Do not publish this project or deploy with real student records to an untrusted audience.** For production, replace codes with Microsoft Entra ID/UTP single sign-on, server-side authorization and audit logs. Do not commit `data/input/` publicly.

The updated workbook assigns all **97** students to four lecturers: **Noureen Talpur (30)**, **Jasmin (28)**, **Azhar (22)**, and **Nurlaili (17)**. Lecturer codes are distinct six-character alphanumeric strings, shared by all students assigned to the same lecturer. All student and lecturer email fields are populated in the supplied workbook. The app reads this workbook each time it reloads the contacts cache; code-based access limits lecturers to their own students.

## Dashboard section order (matches DOCX wireframes)

### Admin: Overall Cohort

**Header + Home** → Last Updated → **six KPI cards** (Students, Average GPA, Average CGPA, Students Predicted At-Risk, At-Risk Rate, Courses Failed) → side-by-side **Student Distribution / Programme Performance (GPA/CGPA)** → **Programme Performance Overview** → three-column **At-Risk Predictions / Academic Standing / Credit Hour Status** → **Overall Individual Student Performance**.

A left sidebar provides Overall/Individual navigation and programme filtering (All, IT, CS, IS).

### Admin: Individual Student

**Header + Home** → Last Updated → three searchable selectors (Student Name, Student ID, Academic Advisor) → six KPI cards → **historical GPA/CGPA and Ridge-regression forecast** → risk threshold bar (24%) beside the **credit-hour donut** → **Semester Timeline Record** (minimum Semester 1–10, extending to 11/13 or further when records exist) → clickable semester tiles with current semester clearly marked → **Subjects by Semester** with default curriculum expander → side-by-side **Failed Courses / Repeat and Incomplete Courses** → **Alignment chart + full timeline table** → **Generate Next Semester** → recommended course table and applicable manual-review/deferred-requirements expanders.

Student portal displays exactly this individual content, with read-only Student Name, Student ID and Advisor text fields replacing the search selectors. Lecturer individual dashboard uses name/ID selectors limited to assigned students.

### Academic Advisor: Overall Cohort

Same cohort KPI/header layout, with an additional **Student Overview** table directly below KPIs. Selecting its row opens that student's individual advisor dashboard. All charts/tables and selectors are server-side filtered to the lecturer's assigned students.

## Data and ML details

- `data/input/S1_S97_Centralised_Academic_PO_Workbook.xlsx`: updated student names/IDs, semester history, course attempts, enrolment and programme outcomes.
- `data/input/IT_CS_IS_Semester_Curriculum.xlsx`: updated IT/CS/IS curriculum, default semester, credit hours and prerequisites.
- `data/input/Student_Lecturer_DB.xlsx`: new student access codes, names, IDs, emails, lecturer mappings and lecturer access codes.
- `models/risk_model.joblib`: existing random-forest early-warning model, decision threshold approximately **23.99%** (displayed as 24%). The replacement academic workbook changes matric identifiers/names without changing the historical model features and labels, so the existing model remains usable. Retrain after substantive academic updates.
- GPA/CGPA forecast: **student-specific Ridge regression**, fitted only on available historical semester grades. Forecast lines are visually distinct from recorded grades and separated by a vertical divider; 2.0 reference line is horizontal. Forecasts are exploratory, not guaranteed grade outcomes and not produced by the risk classifier.
- Suggested retake/incomplete-course actions combine academic rules and the ML risk estimate. The risk classifier **does not generate free-text actions**, and the system should not be described as an ML action-generation model.
- Credit-hour charts count curriculum-mapped course requirements, with the *latest* recorded attempt determining passed or failed status; repeated attempts do not inflate completed hours. Unmapped/legacy credit-hour coverage is explicitly noted. Targets: IT/IS **122**, CS **123**.
- GPA/CGPA averages use each student's latest recorded semester. `F` and `QF` both count as failed attempts. Students without SAS history remain on cohort lists but have no risk prediction.

## SIT/SIP constraints

The pathway engine uses the student's **actual** semester of SIT/SIP rather than requiring their default curriculum date. SIT and SIP each occupy an exclusive semester. After SIT, the very next semester is reserved solely for SIP; a currently enrolled SIT gives a *conditional* SIP recommendation subject to its successful completion. If neither is taken and the SIT default semester has passed, the next planned term reserves SIT instead. Historical data with nonconsecutive SIT/SIP or additional subjects in an industrial term are flagged and **not altered**. Default SIT/SIP entries remain visible in the curriculum expander.

## Edits3: future pathway and academic chatbot

Under **Recommended Future Semester Courses**, adjust **Maximum credit hours for future semesters** with the +/- input; it changes future terms without altering the already-generated next term. Select a future semester using the dropdown. Projection stops when the required passing credits have been modelled, no further academic-credit progress is possible, or it would repeat the same requirements. SIT/SIP terms keep their mandatory single-course placement even when this exceeds a reduced user-selected future limit. All projected passes are hypothetical and require advisor confirmation.

The **Academic Dashboard Chatbot** appears in Admin, Lecturer and Student views. It now includes **Suggested questions**, a dropdown and an **Ask question** button, alongside free-text chat input. Both **Generative AI** and **local/offline** modes have separately curated questions for individual-student and cohort dashboards. Each suggested question has a targeted fact-based offline response (including when the optional AI service is unavailable), and the two modes maintain separate chat histories. The default **Generative AI: Off** mode answers questions using the currently authorised dashboard figures and semester/course history without network/API access. For optional generative responses, set the `OPENAI_API_KEY` environment variable before launching Streamlit; optionally set `OPENAI_MODEL` (default: `gpt-4o-mini`), then turn **Generative AI** on in the dashboard. For example, in PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
python -m streamlit run app.py
```

When enabled, the chatbot sends de-identified academic dashboard context and the user's questions to the configured OpenAI API. Only students in the logged-in role's current view are included. Leave it off if external AI data processing is not permitted by institutional policy; never commit API keys to the project ZIP. If no key is configured or the API fails, the rule-based dashboard chatbot remains available.

## Files and tests

- `app.py`: role-based Streamlit UI and wireframe section ordering
- `src/portal_data.py`: auth lookup, forecasting, credit calculations and academic tables
- `src/industrial_rules.py`: dynamic SIT/SIP rules, validation
- `src/pathway_engine.py`: existing optimiser adapted for exclusive, student-specific industrial semesters and study extensions past Semester 10
- `tests/test_portal.py`: offline regression tests
- `IMPLEMENTATION_NOTES.md`: further handover and limitations
- `docs/AI Prompt.docx`: the document supplied as the acceptance specification

From the project root:

```powershell
python -m unittest discover -s tests -v
```

**Deployment limitation:** The implementation environment did not have Streamlit installed and cannot fetch it, so app launch and pixel-level browser rendering were not verified here. Python compilation, data regeneration, ML snapshot and pathway/regression tests were verified. Streamlit/Plotly will be installed from `requirements.txt` on a connected local machine.

## Visual refresh, themes and home background

- The supplied campus photograph is included at `assets/UTP.jpeg` and embedded on the home screen locally. No external image host is necessary.
- The full-width university header and centered **Academic Dashboard** hero appear above the Student/Lecturer entry buttons.
- KPI cards, chart containers, data tables, chat controls and navigation have updated spacing, rounded surfaces and stronger visual hierarchy. Dashboard surfaces and text inherit Streamlit's light/dark theme variables rather than forcing white backgrounds or dark text. Plotly charts inherit the active Streamlit theme and use transparent plot backgrounds.
- Switch between Light and Dark via Streamlit's top-right **Settings → Theme** menu. The UTP campus home screen intentionally uses a dark photographic overlay and high-contrast white type in both themes.
- Existing ML and processed academic tables have not been changed: updated lecturer mapping is read from the replacement contacts workbook.

Use `python -m unittest discover -s tests -p 'test_chat_prompts.py' -v` to run the new dashboard-question and lecturer-access checks even without the optional OR-Tools pathway dependency. The full suite requires all packages in `requirements.txt`. Streamlit 1.51+ is used for coordinated selectable light and dark theme configuration in `.streamlit/config.toml`.
