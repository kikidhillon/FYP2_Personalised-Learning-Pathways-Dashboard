# Continuation prompt — UTP Personalised Learning Academic Dashboard

This repository contains the updated, role-based Streamlit project based on `docs/AI Prompt.docx`. Use **FYP_Personalised_Learning_2_Updated.zip** as the working project. The three current source workbooks in `data/input/` are the source of truth for student records, IT/CS/IS curriculum, and Student/Lecturer assignments. The current data identify students by their supplied names/IDs (not anonymised data).

Admin: all students; Student: own individual dashboard via Student_Code; Academic Advisor: only assigned students via Lecturer_Code. The model is a random-forest early-risk predictor; individual GPA/CGPA forecasts use separate student-specific Ridge regression. SIT/SIP exclusive semesters follow actual recorded placement and SIP immediately follows SIT. The project retains prerequisite-checked next-semester pathways and student curriculum alignment.

See README.md and IMPLEMENTATION_NOTES.md for execution, dashboard placements, access-code limitations, workbook mappings and verification. Run `python -m unittest discover -s tests -v` and `python -m streamlit run app.py` after installing requirements.
