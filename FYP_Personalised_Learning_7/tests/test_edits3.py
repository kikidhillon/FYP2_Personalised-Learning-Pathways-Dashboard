"""Regression tests for edits3: future projection controls and scoped dashboard Q&A."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.dashboard_chat import build_dashboard_context, answer_dashboard_question
from src.pathway_engine import generate_future_pathway, generate_pathway

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'


class Edits3Tests(unittest.TestCase):
    def test_future_limit_changes_only_future_semesters(self):
        initial=generate_pathway('S1',16,DATA)
        self.assertTrue(initial['summary']['fixed_industrial_semester'])
        self.assertEqual(initial['summary']['maximum_credits'],7)
        full=generate_future_pathway('S1',7,DATA,initial=initial,future_maximum_credits=16)
        reduced=generate_future_pathway('S1',7,DATA,initial=initial,future_maximum_credits=6)
        for cap, outcome in ((16,full),(6,reduced)):
            by_term=outcome['courses'].groupby('Semester Number')['Credit Hours'].sum()
            self.assertTrue((by_term <= cap).all())
            self.assertFalse(outcome['incomplete'])
            self.assertEqual(outcome['passed_credits_planned'],122)
            self.assertEqual(len(by_term),len(by_term.index.unique()))
        self.assertLess(len(full['courses']['Semester Number'].unique()),
                        len(reduced['courses']['Semester Number'].unique()))
        self.assertEqual(initial['summary']['maximum_credits'],7)

    def test_repeated_no_credit_progress_stops_without_spurious_semesters(self):
        initial=generate_pathway('S1',16,DATA)
        fake={'complete':False,'passed':79.0}
        with patch('src.pathway_engine.credit_status',return_value=fake) as status:
            plan=generate_future_pathway('S1',7,DATA,initial=initial)
        self.assertLess(status.call_count,10)
        self.assertTrue(plan['incomplete'])
        self.assertTrue(plan['courses'].empty)
        self.assertLessEqual(plan['last_semester'],plan['next_semester'])

    def test_offline_chat_scope_and_semester_answers(self):
        ml=pd.read_csv(DATA/'ml_semester.csv')
        alignment=pd.read_csv(DATA/'student_timeline_alignment.csv')
        curriculum=pd.read_csv(DATA/'curriculum_timeline.csv')
        s1=ml[ml.student_key.eq('S1')].sort_values('semester_seq').iloc[-1]
        visible=pd.DataFrame([{'student_key':'S1','programme_code':'IT',
            'gpa':s1.gpa,'cgpa':s1.cgpa,'standing':s1.standing,
            'current_semester':s1.semester_seq,'risk_probability':0.1,
            'predicted_at_risk':False,'failed_courses_all':2}])
        context=build_dashboard_context(visible,ml,alignment,curriculum,
                                        student_key='S1',dashboard='Individual Student')
        self.assertEqual(context['student_count'],1)
        self.assertEqual(context['student']['programme'],'IT')
        self.assertIn('Semester 3',answer_dashboard_question('What happened in semester 3?',context))
        self.assertIn('passing credits',answer_dashboard_question('How many credit hours remain?',context))
        self.assertNotIn('student_name',json.dumps(context))
        self.assertNotIn('matric_number',json.dumps(context))
        inaccessible=build_dashboard_context(visible,ml,alignment,curriculum,student_key='S2')
        self.assertNotIn('student',inaccessible)

if __name__=='__main__':
    unittest.main()
