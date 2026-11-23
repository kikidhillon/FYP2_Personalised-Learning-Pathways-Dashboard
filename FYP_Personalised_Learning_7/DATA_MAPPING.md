# Database and Curriculum Mapping

## Central database worksheets used

| Worksheet | Use |
|---|---|
| `Students` | Programme mapping and latest student summary |
| `ML_Semester` | Risk-model training and prediction |
| `SAS_Semesters` | Semester sequence and academic history |
| `SAS_Courses` | Completed, failed and repeated courses |
| `Current_Enrollment` | Courses that must not be recommended twice |
| `PO_Achievement` | Monitoring and future explainability work |
| `PO_Courses` | Programme-outcome/course relationships |

## Curriculum worksheets used

| Programme | Timeline | Minor electives | Core specialisations |
|---|---|---|---|
| IT | `IT_Timeline` | `IT_EM` | `IT_CSp` |
| CS | `CS_Timeline` | `CS_EM` | `CS_CSp` |
| IS | `IS_Timeline` | `IS_EM` | `IS_CSp` |

## Programme mapping

| Database programme | Curriculum code |
|---|---|
| Bachelor of Information Technology (Hons) | IT |
| Bachelor of Computer Science (Hons) | CS |
| Bachelor of Information Systems (Hons) | IS |

## Alignment labels

- `On default semester`: actual and default semester numbers are equal.
- `Taken earlier`: the course was recorded before its current default semester.
- `Taken later`: the course was recorded after its current default semester.
- `Repeat attempt`: the student attempted the same recorded course again.
- `Unmapped to current curriculum`: no safe code or normalised-name match was found.

Historical codes are matched by course name only when the normalised names are
equal within the student's programme. The code does not use fuzzy matching,
because a false curriculum match is more harmful than leaving a row for manual
review.


## New student/lecturer directory (2026 update)

`data/input/Student_Lecturer_DB.xlsx` / sheet `DB`:

| Source column | Used for |
|---|---|
| `Student_Key` | Exact join to central student and semester tables |
| `Matric_Number` | Current university student identifier |
| `Student_Name` | Human-readable dashboard name |
| `Programme` | IT/CS/IS lookup |
| `Student_UTP_Email` | Read-only student contact information |
| `Student_Code` | Prototype single-student access lookup |
| `Lecturer` | Academic advisor name |
| `Lecturer_UTP_Email` | Read-only advisor contact |
| `Lecturer_Code` | Prototype advisor access lookup and authorized student scope |

The replacement directory assigns all 97 students to four lecturers: Noureen Talpur (30), Jasmin (28), Azhar (22), and Nurlaili (17). All student and lecturer email fields are populated. Each lecturer code grants access only to that lecturer's assigned students; admin and individual student-code access still follow the existing scope checks.
