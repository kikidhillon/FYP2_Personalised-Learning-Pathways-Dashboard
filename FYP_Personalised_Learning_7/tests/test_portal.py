"""Portable smoke/regression checks that do not require Streamlit."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.dashboard_data import build_student_snapshot,load_model_bundle
from src.industrial_rules import SIT,SIP,industrial_record,planned_industrial_course,validate_industrial_history
from src.pathway_engine import generate_pathway
from src.portal_data import (
    REQUIRED_CREDITS,auth_from_code,authorized_keys,credit_status,enrich_roster,
    forecast_academic_trend,latest_update,load_directory,semester_label,student_timeline,
)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'

class PortalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=load_directory(ROOT/'data'/'input'/'Student_Lecturer_DB.xlsx')
        cls.students=pd.read_csv(DATA/'students.csv')
        cls.ml=pd.read_csv(DATA/'ml_semester.csv')
        cls.courses=pd.read_csv(DATA/'student_timeline_alignment.csv')
        cls.curriculum=pd.read_csv(DATA/'curriculum_timeline.csv')
        cls.enrollment=pd.read_csv(DATA/'current_enrollment_alignment.csv')
        cls.model=load_model_bundle(ROOT/'models'/'risk_model.joblib')

    def test_directory_exact_access_codes(self):
        self.assertEqual(len(self.directory),97)
        self.assertEqual(auth_from_code(self.directory,'admin','adminUTP97'),'admin')
        self.assertIsNone(auth_from_code(self.directory,'admin','ADMINUTP97'))
        self.assertEqual(auth_from_code(self.directory,'student','k3D4p2'),'S1')
        self.assertIsNone(auth_from_code(self.directory,'student','incorrect'))
        self.assertEqual(len(authorized_keys(self.directory,'student','S1')),1)
        self.assertEqual(len(authorized_keys(self.directory,'lecturer','A1B2C3')),30)
        self.assertEqual(len(authorized_keys(self.directory,'admin','admin')),97)
        self.assertNotIn('S97',authorized_keys(self.directory,'lecturer','A1B2C3'))

    def test_new_workbooks_rebuilt(self):
        self.assertEqual(len(self.students),97)
        self.assertEqual(len(self.ml),665)
        self.assertEqual(len(self.courses),2893)
        self.assertEqual(self.students.iloc[0]['student_name'],'Kieron Dimaz Singh Dhillon')
        self.assertEqual(latest_update(self.ml),'May 2026')
        self.assertEqual(self.curriculum.groupby('programme_code').credit_hours.sum().to_dict(),REQUIRED_CREDITS)

    def test_ml_coverage_and_roster(self):
        snap=build_student_snapshot(self.students,self.ml,self.model)
        self.assertEqual(len(snap),97)
        self.assertEqual(int(snap.risk_probability.notna().sum()),96)
        snap['current_semester']=snap.semester_seq
        enhanced=enrich_roster(snap,self.directory,self.ml,pd.read_csv(DATA/'sas_courses.csv'),self.enrollment)
        self.assertEqual(len(enhanced),97)
        self.assertEqual(enhanced.loc[enhanced.student_key.eq('S1'),'matric_number'].iloc[0],'22005410')

    def test_semester_13_and_forecast(self):
        latest=self.ml.groupby('student_key').semester_seq.max()
        self.assertGreaterEqual(latest.max(),13)
        key=latest.idxmax()
        h=self.ml[self.ml.student_key.eq(key)]
        a=self.courses[self.courses.student_key.eq(key)]
        e=self.enrollment[self.enrollment.student_key.eq(key)]
        timeline=student_timeline(h,e,a)
        self.assertGreaterEqual(len(timeline),13)
        forecast=forecast_academic_trend(h)
        self.assertGreaterEqual(int(forecast.Semester.max()),15)
        self.assertTrue(forecast.loc[forecast.Semester.le(13),'Historical CGPA'].notna().any())
        self.assertTrue(forecast.loc[forecast.Semester.gt(13),'Forecast CGPA'].notna().any())

    def test_credits_not_exaggerated_by_repeated_attempts(self):
        for key in ['S1','S2','S60']:
            programme=self.students.loc[self.students.student_key.eq(key),'programme_code'].iloc[0]
            x=credit_status(self.courses[self.courses.student_key.eq(key)],self.curriculum,programme)
            self.assertEqual(x['required'],REQUIRED_CREDITS[programme])
            self.assertAlmostEqual(x['passed']+x['failed']+x['remaining'],x['required'])
            self.assertTrue(all(v>=0 for v in [x['passed'],x['failed'],x['remaining']]))

    def test_industrial_actual_placement_overrides_default(self):
        h=pd.DataFrame({'semester_seq':[9],'module_code':['IFB3037'],'module':['Student Industrial Training']})
        empty=h.iloc[0:0]
        self.assertEqual(industrial_record(h,empty),{'SIT':9})
        fixed=planned_industrial_course(self.curriculum,'IT',10,h,empty)
        self.assertEqual(fixed.iloc[0].course_name,SIP)
        self.assertEqual(len(fixed),1)
        self.assertFalse(planned_industrial_course(self.curriculum,'IT',10,empty,empty).empty)
        self.assertEqual(planned_industrial_course(self.curriculum,'IT',10,empty,empty).iloc[0].course_name,SIT)
        h2=pd.concat([h,pd.DataFrame({'semester_seq':[10],'module_code':['IFB3047'],'module':['Student Industrial Project']})])
        self.assertTrue(planned_industrial_course(self.curriculum,'IT',11,h2,empty).empty)
        invalid=pd.concat([h,pd.DataFrame({'semester_seq':[9],'module_code':['TFB1013'],'module':['Programming']})])
        self.assertTrue(validate_industrial_history(invalid))

    def test_real_next_semester_pathway_and_exclusivity(self):
        result=generate_pathway('S1',24,DATA)
        self.assertEqual(result['summary']['next_default_semester'],9)
        self.assertTrue(result['summary']['additional_courses_locked'])
        self.assertEqual(result['recommendations'].course_name.tolist(),[SIP])
        self.assertEqual(result['summary']['maximum_credits'],7)
        # A student enrolled in SIT at Semester 7 must receive SIP-only in Semester 8.
        enrolled=generate_pathway('S60',24,DATA)
        self.assertEqual(enrolled['summary']['next_default_semester'],8)
        self.assertEqual(enrolled['recommendations'].course_name.tolist(),[SIP])
        self.assertIn('Conditional:',enrolled['recommendations'].iloc[0].prerequisite_status)

if __name__=='__main__':unittest.main()
