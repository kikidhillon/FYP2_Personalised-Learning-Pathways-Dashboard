"""Student-specific SIT/SIP sequence constraints, independent of curriculum defaults."""
from __future__ import annotations

import pandas as pd

SIT = 'Student Industrial Training (SIT)'
SIP = 'Student Industrial Project (SIP)'
CODES = {'IFB3037':'SIT','IEB3037':'SIT','IFB3047':'SIP','IEB3047':'SIP'}


def industrial_type(code: object, name: object) -> str | None:
    value = str(code).strip().upper()
    if value in CODES: return CODES[value]
    name = str(name).lower()
    if 'student industrial training' in name: return 'SIT'
    if 'student industrial project' in name: return 'SIP'
    return None


def industrial_record(history: pd.DataFrame, current: pd.DataFrame) -> dict[str, int]:
    """Actual SIT/SIP semester takes precedence over catalogue default."""
    out={}
    for df in [history,current]:
        if df.empty: continue
        for row in df.itertuples():
            kind=industrial_type(getattr(row,'module_code',''),getattr(row,'module',''))
            n=getattr(row,'semester_seq',None)
            if kind and pd.notna(n): out[kind]=int(n)
    return out


def planned_industrial_course(timeline: pd.DataFrame, programme: str, next_semester: int,
                              history: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Return the exclusive industrial course for the *student's* next semester.

    SIT/SIP default slots are a guideline only. Once SIT actually occurs, the
    immediately following semester is reserved solely for SIP. If SIT has been
    delayed beyond its default date, the next study semester reserves SIT.
    """
    programme_rows=timeline[timeline.programme_code.eq(programme)]
    industrial=programme_rows[programme_rows.course_name.isin([SIT,SIP])]
    actual=industrial_record(history,current)
    sit_default=industrial.loc[industrial.course_name.eq(SIT),'default_semester']
    if sit_default.empty: return industrial.iloc[0:0]
    sit_due=int(sit_default.iloc[0])
    if 'SIP' in actual: return industrial.iloc[0:0]
    if 'SIT' in actual:
        if next_semester > actual['SIT']:
            return industrial[industrial.course_name.eq(SIP)].copy()
        return industrial.iloc[0:0]
    if next_semester >= sit_due:
        return industrial[industrial.course_name.eq(SIT)].copy()
    return industrial.iloc[0:0]


def validate_industrial_history(history: pd.DataFrame) -> list[str]:
    alerts=[]
    if history.empty: return alerts
    for sem,group in history.groupby('semester_seq'):
        kinds=[industrial_type(r.module_code,r.module) for r in group.itertuples()]
        if any(kinds) and len(group)>1:
            alerts.append(f'Semester {sem} contains an industrial course and {len(group)-1} additional course attempt(s). Source records were not changed.')
    actual=industrial_record(history,history.iloc[0:0])
    if 'SIP' in actual and 'SIT' in actual and actual['SIP'] != actual['SIT']+1:
        alerts.append(f"SIT in Semester {actual['SIT']} and SIP in Semester {actual['SIP']} are not consecutive in the supplied records.")
    return alerts
