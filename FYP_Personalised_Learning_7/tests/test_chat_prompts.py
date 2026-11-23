"""Verify real directory authorization and distinct fact-grounded suggested Q&A."""
import unittest
from pathlib import Path

import pandas as pd

from src.dashboard_chat import (answer_dashboard_question, build_dashboard_context,
                                suggested_prompts)
from src.portal_data import auth_from_code, authorized_keys, load_directory

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data' / 'processed'


class DashboardRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = load_directory(ROOT / 'data' / 'input' / 'Student_Lecturer_DB.xlsx')
        cls.ml = pd.read_csv(PROCESSED / 'ml_semester.csv')
        cls.courses = pd.read_csv(PROCESSED / 'student_timeline_alignment.csv')
        cls.curriculum = pd.read_csv(PROCESSED / 'curriculum_timeline.csv')

    def test_directory_has_97_students_and_unique_lecturer_codes(self):
        frame = self.directory
        self.assertEqual(len(frame), 97)
        self.assertEqual(frame.student_key.nunique(), 97)
        self.assertFalse(frame.student_utp_email.isna().any())
        self.assertFalse(frame.lecturer_utp_email.isna().any())
        expected = {'Noureen Talpur': 30, 'Jasmin': 28, 'Azhar': 22, 'Nurlaili': 17}
        self.assertEqual(frame.lecturer.value_counts().to_dict(), expected)
        self.assertEqual(frame.lecturer_code.nunique(), 4)
        for lecturer, group in frame.groupby('lecturer'):
            code = group.lecturer_code.iloc[0]
            self.assertEqual(len(code), 6)
            self.assertTrue(code.isalnum())
            self.assertEqual(auth_from_code(frame, 'lecturer', code), code)
            self.assertEqual(len(authorized_keys(frame, 'lecturer', code)), len(group))

    def test_prompts_differ_by_scope_and_mode_and_have_distinct_answers(self):
        one = self.ml[self.ml.student_key.eq('S1')].sort_values('semester_seq').iloc[-1]
        person = pd.DataFrame([dict(student_key='S1', programme_code='IT',
            gpa=one.gpa, cgpa=one.cgpa, standing=one.standing,
            current_semester=one.semester_seq, risk_probability=0.1,
            predicted_at_risk=False, failed_courses_all=2)])
        individual = build_dashboard_context(person, self.ml, self.courses,
            self.curriculum, student_key='S1', dashboard='Individual Student')
        students = []
        for student_key, rows in self.ml.groupby('student_key'):
            r = rows.sort_values('semester_seq').iloc[-1]
            students.append(dict(student_key=student_key, programme_code=r.programme_code,
                gpa=r.gpa, cgpa=r.cgpa, standing=r.standing,
                current_semester=r.semester_seq, risk_probability=0.1,
                predicted_at_risk=False, failed_courses_all=0))
        cohort = build_dashboard_context(pd.DataFrame(students), self.ml,
            self.courses, self.curriculum)
        for individual_mode, context in ((True,individual),(False,cohort)):
            local = suggested_prompts(individual=individual_mode,generative=False)
            ai = suggested_prompts(individual=individual_mode,generative=True)
            self.assertNotEqual(local,ai)
            for mode_prompts in (local,ai):
                self.assertGreaterEqual(len(mode_prompts), 9)
                answers = [answer_dashboard_question(q,context) for q in mode_prompts]
                self.assertEqual(len(set(answers)), len(answers))
                self.assertTrue(all(len(answer)>35 for answer in answers))
        # Names, matric numbers and codes never enter the optional API's context.
        self.assertNotIn('student_name', str(individual))
        self.assertNotIn('matric_number', str(individual))

    def test_student_scope_cannot_show_another_students_history(self):
        student = pd.DataFrame([dict(student_key='S1', programme_code='IT',gpa=2.9,
            cgpa=3.1,standing='Good Standing',current_semester=3,
            risk_probability=0.1,predicted_at_risk=False,failed_courses_all=1)])
        context = build_dashboard_context(student,self.ml,self.courses,self.curriculum,
                                          student_key='S2',dashboard='Individual Student')
        self.assertNotIn('student',context)

    def test_campus_background_asset_exists(self):
        self.assertTrue((ROOT / 'assets' / 'UTP.jpeg').is_file())


if __name__ == '__main__':
    unittest.main()
