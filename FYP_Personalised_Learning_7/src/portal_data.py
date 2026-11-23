"""Role mapping, semester data, academic summaries and display-safe data for the dashboards."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.data_access import normalise_columns
from src.dashboard_data import PROGRAMME_LABELS

REQUIRED_CREDITS = {'IT': 122, 'IS': 122, 'CS': 123}
FAIL = {'QF', 'F'}
PERIODS = ['January', 'May', 'September']


def load_directory(path: Path) -> pd.DataFrame:
    frame = normalise_columns(pd.read_excel(path, sheet_name='DB', dtype={'Matric_Number': str, 'Student_Code': str, 'Lecturer_Code': str}))
    required = {'student_key','matric_number','student_name','programme','student_code','lecturer','lecturer_code','student_utp_email','lecturer_utp_email'}
    if missing := required - set(frame.columns):
        raise ValueError(f'Student_Lecturer_DB is missing: {sorted(missing)}')
    for col in ['student_key','student_code','lecturer_code','lecturer','student_utp_email','lecturer_utp_email']:
        frame[col] = frame[col].astype('string').str.strip()
    if frame['student_key'].duplicated().any() or frame['student_code'].duplicated().any():
        raise ValueError('Student directory keys/access codes must be unique.')
    return frame


def authorized_keys(directory: pd.DataFrame, role: str, principal: str) -> set[str]:
    if role == 'admin':
        return set(directory.student_key.dropna().astype(str))
    if role == 'student':
        return set(directory.loc[directory.student_key.eq(principal), 'student_key'].astype(str))
    if role == 'lecturer':
        return set(directory.loc[directory.lecturer_code.eq(principal), 'student_key'].astype(str))
    return set()


def auth_from_code(directory: pd.DataFrame, role: str, code: str) -> str | None:
    """Return canonical principal, never trust a client-submitted student key."""
    if role == 'admin':
        import hmac
        return 'admin' if hmac.compare_digest(code, 'adminUTP97') else None
    column, return_column = ('student_code','student_key') if role == 'student' else ('lecturer_code','lecturer_code')
    hits = directory.loc[directory[column].eq(code)]
    return str(hits.iloc[0][return_column]) if not hits.empty else None


def period_month(value: object) -> str:
    p = str(value).lower()
    for name in PERIODS:
        if name.lower() in p:
            return name
    return ''


def latest_update(ml: pd.DataFrame) -> str:
    valid = ml.copy()
    valid['year'] = pd.to_numeric(valid['year'], errors='coerce')
    valid['_month'] = valid['period'].apply(lambda p: {'January':1,'May':5,'September':9}.get(period_month(p),0))
    valid = valid.dropna(subset=['year']).sort_values(['year','_month'])
    return 'Not available' if valid.empty else f"{period_month(valid.iloc[-1]['period'])} {int(valid.iloc[-1]['year'])}"


def next_period(year: int, period: str) -> tuple[int,str]:
    month = period_month(period)
    if month == 'January': return year, 'May Semester'
    if month == 'May': return year, 'September Semester'
    return year+1, 'January Semester'


def semester_label(seq: object, period: object, year: object) -> str:
    if pd.isna(seq): return 'N/A'
    s = int(seq)
    if pd.isna(year) or not period_month(period): return str(s)
    return f'{s} ({period_month(period)} {int(year)})'



def chronological_enrollment(ml: pd.DataFrame, enrollment: pd.DataFrame) -> pd.DataFrame:
    """Reject duplicate/earlier enrollment periods masquerading as later semesters.

    Student_SAS Semester_Seq is authoritative for a period already recorded.
    Enrollment advances it only when the enrollment's Year/Period are later.
    """
    if enrollment.empty or ml.empty:
        return enrollment.copy()
    latest = ml.sort_values('semester_seq').drop_duplicates('student_key', keep='last')
    latest = latest[['student_key', 'year', 'period']].rename(columns={'year':'last_year','period':'last_period'})
    joined = enrollment.merge(latest, on='student_key', how='left')
    months = {'January':1,'May':5,'September':9}
    enrolled_dates = (pd.to_numeric(joined.year,errors='coerce')*12
                      + joined.period.map(lambda v: months.get(period_month(v),0)))
    recorded_dates = (pd.to_numeric(joined.last_year,errors='coerce')*12
                      + joined.last_period.map(lambda v: months.get(period_month(v),0)))
    valid = recorded_dates.isna() | (enrolled_dates > recorded_dates)
    return joined.loc[valid, enrollment.columns].reset_index(drop=True)


def course_credit_hours(rows: pd.DataFrame, curriculum: pd.DataFrame, programme: str,
                        options: pd.DataFrame | None = None) -> pd.Series:
    """Look up course credits from curriculum mappings, without guessing unknown hours."""
    from src.data_access import normalise_code
    catalog = curriculum[curriculum.programme_code.eq(programme)]
    if options is not None and not options.empty:
        catalog = pd.concat([catalog, options[options.programme_code.eq(programme)]], ignore_index=True)
    by_id = catalog.drop_duplicates('curriculum_row_id').set_index('curriculum_row_id').credit_hours.to_dict()
    code_map = {}
    for r in catalog.itertuples():
        for code in str(r.course_code_raw).split('/'):
            token = normalise_code(code)
            if token and not token.startswith('SELECT_'):
                code_map[token] = r.credit_hours
    values = []
    for r in rows.itertuples():
        row_id = getattr(r, 'curriculum_row_id', None)
        code = getattr(r,'module_code', getattr(r,'course_code_raw',None))
        value = by_id.get(row_id)
        if pd.isna(value) if value is not None else True:
            value = code_map.get(normalise_code(code))
        values.append(pd.to_numeric(value,errors='coerce') if value is not None else float('nan'))
    return pd.Series(values, index=rows.index)

def credit_status(history: pd.DataFrame, curriculum: pd.DataFrame, programme: str) -> dict[str, float]:
    """Credit allocation by latest attempt of each mapped curriculum course.

    Repeated attempts do not inflate earned or failed hours. Unmapped rows have
    unknown curriculum credit values and are intentionally excluded.
    """
    required = REQUIRED_CREDITS.get(programme, 122)
    if history.empty:
        return {'passed':0.,'failed':0.,'remaining':float(required),'credits_to_pass':float(required),'required':required,'unmapped':0,'complete':False}
    credits = curriculum[['curriculum_row_id','credit_hours']].drop_duplicates('curriculum_row_id')
    mapped = history.dropna(subset=['curriculum_row_id']).copy()
    mapped = mapped.sort_values(['semester_seq','attempt_number']).drop_duplicates('curriculum_row_id',keep='last')
    mapped = mapped.merge(credits,on='curriculum_row_id',how='left')
    mapped['credit_hours'] = pd.to_numeric(mapped['credit_hours'],errors='coerce').fillna(0)
    failed = mapped.grade.astype(str).str.upper().str.strip().isin(FAIL)
    passed = float(mapped.loc[~failed,'credit_hours'].sum())
    fail_hours = float(mapped.loc[failed,'credit_hours'].sum())
    # Curriculum selection slots and unmapped historic courses may mean earned
    # credits are an underestimate. Do not mislabel the remainder as verified.
    remaining = max(0.,float(required)-passed-fail_hours)  # Yet-unattempted; failed hours also remain to be passed.
    return {'passed':passed,'failed':fail_hours,'remaining':remaining,'required':required,
            'credits_to_pass':max(0.,float(required)-passed),
            'unmapped':int(history.curriculum_row_id.isna().sum()),'complete':passed>=required}


def student_timeline(history: pd.DataFrame, enrollment: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    enrollment = chronological_enrollment(history, enrollment)
    newest_history = history.sort_values('semester_seq').drop_duplicates('semester_seq', keep='last')
    by_sem = {int(r.semester_seq):r for r in newest_history.itertuples()}
    enrols = {int(v) for v in pd.to_numeric(enrollment.get('semester_seq',pd.Series(dtype=float)),errors='coerce').dropna()}
    highest = max(10, max(by_sem,default=0), max(enrols,default=0))
    newest = max(max(by_sem,default=0),max(enrols,default=0))
    failures = courses[courses.grade.astype(str).str.upper().str.strip().isin(FAIL)].groupby('semester_seq').size().to_dict() if not courses.empty else {}
    rows=[]
    for n in range(1,highest+1):
        r=by_sem.get(n)
        if r is not None:
            period,year,gpa,cgpa,standing = r.period,r.year,r.gpa,r.cgpa,r.standing
        elif n in enrols and not enrollment[enrollment.semester_seq.eq(n)].empty:
            s=enrollment[enrollment.semester_seq.eq(n)].iloc[-1]
            period,year,gpa,cgpa,standing=s.get('period'),s.get('year'),np.nan,np.nan,'Not yet recorded'
        else:
            period,year,gpa,cgpa,standing='',np.nan,np.nan,np.nan,''
        status = 'Current' if n==newest else 'Recorded' if r is not None else 'Unrecorded'
        rows.append({'Semester Number':n,'Semester': f'{period_month(period)} Semester' if period_month(period) else '—',
                     'Year': int(year) if pd.notna(year) else None,'Status':status,
                     'GPA':gpa, 'CGPA':cgpa,'Standing':standing if pd.notna(standing) else '—',
                     'Failed Courses':int(failures.get(n,0))})
    return pd.DataFrame(rows)


def forecast_academic_trend(history: pd.DataFrame, total_credits: float | None = None) -> pd.DataFrame:
    """Per-student regularized regression forecast: descriptive, not risk classifier.

    Ridge is fitted independently for GPA/CGPA; no future grades are used as
    features. Short histories give no projected values.
    """
    if history.empty: return pd.DataFrame(columns=['Semester','Historical GPA','Historical CGPA','Forecast GPA','Forecast CGPA'])
    h=history.sort_values('semester_seq').drop_duplicates('semester_seq',keep='last')
    last=int(h.semester_seq.max())
    n_future=max(0,max(10,last+2)-last)
    all_s=range(1,last+n_future+1)
    idx=h.set_index('semester_seq')
    rows=[]
    for n in all_s:
        rows.append({'Semester':n,'Historical GPA':idx.loc[n,'gpa'] if n in idx.index else np.nan,
                     'Historical CGPA':idx.loc[n,'cgpa'] if n in idx.index else np.nan,
                     'Forecast GPA':np.nan,'Forecast CGPA':np.nan})
    result=pd.DataFrame(rows)
    for field,output in [('gpa','Forecast GPA'),('cgpa','Forecast CGPA')]:
        usable=h[['semester_seq',field]].dropna()
        if len(usable)<3: continue
        x=usable.semester_seq.to_numpy(dtype=float).reshape(-1,1)
        y=usable[field].to_numpy(dtype=float)
        # Recent scores receive greater weighting, regularization reduces overfit.
        m=Ridge(alpha=3.0).fit(x,y,sample_weight=np.linspace(.6,1.5,len(usable)))
        future=result['Semester']>last
        result.loc[future,output]=np.clip(m.predict(result.loc[future,'Semester'].to_numpy().reshape(-1,1)),0,4)
        # Connect the forecast to the last observed value without faking points.
        if last in result.Semester.values:
            result.loc[result.Semester.eq(last),output]=float(usable.iloc[-1][field])
    return result


def enrich_roster(snapshot: pd.DataFrame, directory: pd.DataFrame, ml: pd.DataFrame,
                  courses: pd.DataFrame, enrollment: pd.DataFrame | None = None) -> pd.DataFrame:
    contact=directory[['student_key','student_name','matric_number','student_utp_email','lecturer','lecturer_utp_email']]
    frame=snapshot.drop(columns=[c for c in ['student_name','matric_number'] if c in snapshot.columns]).merge(contact,on='student_key',how='left')
    ml_latest=ml.sort_values('semester_seq').groupby('student_key',as_index=False).tail(1)
    frame=frame.merge(ml_latest[['student_key','period','year']],on='student_key',how='left')
    if enrollment is not None and not enrollment.empty:
        enrollment = chronological_enrollment(ml, enrollment)
        enrolled=enrollment.sort_values('semester_seq').groupby('student_key',as_index=False).tail(1)
        enrolled=enrolled[['student_key','semester_seq','period','year']].rename(columns={
            'semester_seq':'active_enrollment_seq','period':'enrolled_period','year':'enrolled_year'})
        frame=frame.merge(enrolled,on='student_key',how='left')
        ahead=pd.to_numeric(frame.active_enrollment_seq,errors='coerce') > pd.to_numeric(frame.semester_seq,errors='coerce').fillna(0)
        frame.loc[ahead,'period']=frame.loc[ahead,'enrolled_period']
        frame.loc[ahead,'year']=frame.loc[ahead,'enrolled_year']
    frame['current_semester_label']=[semester_label(s,p,y) for s,p,y in zip(frame.current_semester,frame.period,frame.year)]
    fail_counts=courses[courses.grade.astype(str).str.upper().str.strip().isin(FAIL)].groupby('student_key').size()
    frame['failed_courses_all']=frame.student_key.map(fail_counts).fillna(0).astype(int)
    course_counts=courses.groupby('student_key').size()
    frame['courses_taken_all']=frame.student_key.map(course_counts).fillna(0).astype(int)
    return frame


def course_action(grade: str, risk: float | None, prerequisite: bool = False) -> str:
    if prerequisite: return 'Advisor review: verify prerequisite / equivalence'
    if str(grade).upper() in FAIL:
        return 'Retake when eligible; discuss with advisor' if pd.notna(risk) and risk >= .24 else 'Plan repeat subject; confirm prerequisite'
    return 'Review degree requirements and eligibility'
