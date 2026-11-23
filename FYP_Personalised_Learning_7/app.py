"""UTP Academic Dashboard — document-specified admin, student and advisor portals.

Launch: python -m streamlit run app.py
Role codes come from data/input/Student_Lecturer_DB.xlsx (demonstration access codes).
"""
from __future__ import annotations

from pathlib import Path
import base64
from html import escape
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_data import PROGRAMME_LABELS, build_student_snapshot, load_model_bundle
from src.industrial_rules import SIT, SIP, industrial_record, industrial_type, planned_industrial_course, validate_industrial_history
from src.pathway_engine import generate_pathway, generate_future_pathway
from src.dashboard_chat import (build_dashboard_context, answer_dashboard_question,
                                generate_ai_answer, suggested_prompts)
from src.portal_data import (
    FAIL, REQUIRED_CREDITS, auth_from_code, authorized_keys, course_action,
    credit_status, course_credit_hours, chronological_enrollment, enrich_roster, forecast_academic_trend, latest_update,
    load_directory, next_period, period_month, semester_label, student_timeline,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'/'processed'
CONTACTS = ROOT/'data'/'input'/'Student_Lecturer_DB.xlsx'
MODEL = ROOT/'models'/'risk_model.joblib'
CAMPUS_IMAGE = ROOT/'assets'/'UTP.jpeg'
STATS = ['students','ml_semester','sas_semesters','sas_courses','student_timeline_alignment',
         'current_enrollment_alignment','curriculum_timeline','curriculum_catalog','curriculum_options','semester_credit_plan']
COLORS = {'IT':'#1276ba','CS':'#27a7ae','IS':'#d5a43b'}
PASS_COLOR = '#209568'
FAIL_COLOR = '#d95056'
REMAIN_COLOR = '#f1be45'
RISK_LIMIT = .24   # Dashboard decision boundary specified in AI Prompt.docx

st.set_page_config(page_title='UTP Academic Dashboard', page_icon='🎓', layout='wide', initial_sidebar_state='expanded')
st.markdown("""<style>
/* Theme-aware dashboard styling: respect the viewer's Streamlit light/dark theme. */
:root {--utp-blue:#0c5878; --utp-teal:#08a5a6;}
html, body {overflow-x:hidden;}
header[data-testid="stHeader"] {background:transparent;}
[data-testid="stAppViewContainer"] {
  background:var(--st-background-color, var(--background-color, #f4f7fb));
  color:var(--st-text-color, var(--text-color, #16334a));
}
section[data-testid="stSidebar"] {
  background:var(--st-secondary-background-color, var(--secondary-background-color, #eaf1f6));
  border-right:1px solid color-mix(in srgb, var(--st-text-color, var(--text-color, #16334a)) 15%, transparent);
}
.block-container {max-width:1640px;padding-top:1.1rem;padding-bottom:4rem;}
[data-testid="stMainBlockContainer"] {overflow:visible;}
.utp-dashboard-banner {
  display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;
  width:100%;box-sizing:border-box;padding:17px 26px;border-radius:14px;
  background:linear-gradient(110deg,#0c314d,#0b6281 70%,#068991);
  box-shadow:0 7px 22px rgba(0,39,57,.16);color:#fff;
}
.utp-dashboard-banner .utp-university-name {
  color:#fff;font-size:clamp(1rem,1.6vw,1.48rem);font-weight:800;
  line-height:1.35;letter-spacing:.045em;
}
.utp-dashboard-banner .utp-header-label {
  color:#dbf7f5;font-size:.75rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;
}
.utp-page-title {font-size:clamp(1.16rem,1.85vw,1.72rem);font-weight:750;
  line-height:1.35;padding:13px 1px 5px;color:var(--st-text-color,var(--text-color,#16334a));}
/* Use theme-driven surfaces and text on metric cards, charts, tables and forms. */
div[data-testid="stMetric"] {
  background:var(--st-secondary-background-color,var(--secondary-background-color,#fff));
  color:var(--st-text-color,var(--text-color,#16334a));
  border:1px solid color-mix(in srgb,var(--st-text-color,var(--text-color,#16334a)) 16%,transparent);
  border-top:3px solid #169da1;
  border-radius:14px;padding:15px 14px;min-height:128px;min-width:0;
  box-shadow:0 3px 11px rgba(12,50,68,.06);overflow:visible;
}
div[data-testid="stMetricLabel"] {font-size:.85rem;white-space:normal!important;overflow:visible!important;}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
  color:var(--st-text-color,var(--text-color,#16334a));white-space:normal!important;
  overflow-wrap:anywhere;line-height:1.32;
}
div[data-testid="stMetricValue"] {font-size:clamp(1rem,1.45vw,1.7rem);min-width:0;overflow:visible!important;}
div[data-testid="stMetricValue"]>div {white-space:normal!important;
  overflow-wrap:anywhere;line-height:1.2;text-overflow:clip!important;overflow:visible!important;}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
  background:var(--st-secondary-background-color,var(--secondary-background-color,#fff));
  border:1px solid color-mix(in srgb,var(--st-text-color,var(--text-color,#16334a)) 13%,transparent);
  border-radius:14px;overflow:hidden;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color:color-mix(in srgb,var(--st-text-color,var(--text-color,#16334a)) 15%,transparent)!important;
  border-radius:14px;
}
[data-testid="stChatMessage"] {border-radius:12px;}
.stButton button {border-radius:10px;font-weight:650;min-height:2.6rem;transition:transform .15s,box-shadow .15s;}
.stButton button:hover {transform:translateY(-1px);box-shadow:0 4px 12px rgba(8,57,78,.18);}
.stButton button[kind="primary"] {background:#106b82;border:1px solid #106b82;color:#fff;}
.stButton button[kind="primary"]:hover {background:#0b5369;border-color:#0b5369;color:#fff;}
section[data-testid="stSidebar"] h2 {font-size:1.05rem;}
@media(max-width:760px) {
  .block-container {padding-left:1rem;padding-right:1rem;}
  .utp-dashboard-banner {padding:15px 17px;}
  .utp-dashboard-banner .utp-university-name {font-size:1rem;}
  div[data-testid="stMetric"] {min-height:115px;}
}
/* In dark theme, native color variables automatically switch with Streamlit. */
</style>""",unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(signature: tuple) -> dict[str,pd.DataFrame]:
    missing=[x for x in STATS if not (DATA/f'{x}.csv').exists()]
    if missing: raise FileNotFoundError('Missing processed tables: '+', '.join(missing)+'. Run python -m src.prepare_project_data')
    out={x:pd.read_csv(DATA/f'{x}.csv') for x in STATS}
    out['directory']=load_directory(CONTACTS)
    out['current_enrollment_alignment']=chronological_enrollment(out['ml_semester'],out['current_enrollment_alignment'])
    return out

@st.cache_resource(show_spinner=False)
def model():
    return load_model_bundle(MODEL)

try:
    data_signature=tuple((name,(DATA/f'{name}.csv').stat().st_mtime_ns if (DATA/f'{name}.csv').exists() else 0)
                         for name in STATS) + (('contacts',CONTACTS.stat().st_mtime_ns if CONTACTS.exists() else 0),)
    db=load_data(data_signature)
except Exception as exc:
    st.error(f'Data initialization failed: {exc}')
    st.stop()

bundle = model()
snapshot=build_student_snapshot(db['students'],db['ml_semester'],bundle)
enrolled=db['current_enrollment_alignment'].groupby('student_key')['semester_seq'].max().rename('enrollment_semester')
snapshot=snapshot.merge(enrolled,on='student_key',how='left')
snapshot['current_semester']=pd.concat([
    pd.to_numeric(snapshot.semester_seq,errors='coerce'),pd.to_numeric(snapshot.enrollment_semester,errors='coerce')],axis=1).max(axis=1)
roster=enrich_roster(snapshot,db['directory'],db['ml_semester'],db['sas_courses'],db['current_enrollment_alignment'])
updated=latest_update(db['ml_semester'])


def empty_chart(title: str, notice='No data for the selection') -> None:
    st.markdown(f'**{title}**')
    st.info(notice)


def figure(title: str, height:int=320) -> go.Figure:
    f=go.Figure()
    # Transparent plots + Streamlit's Plotly theme retain text/axis contrast
    # when the viewer switches between light and dark modes.
    f.update_layout(title={'text':title,'font':{'size':17}},
                    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40,r=24,t=58,b=45),height=height,
                    font=dict(size=13),legend=dict(orientation='h',y=-.18),hovermode='closest')
    return f


def show_chart(fig:go.Figure) -> None:
    st.plotly_chart(fig,use_container_width=True,theme='streamlit',config={'displayModeBar':False})


def heading(title: str) -> None:
    st.markdown("""<div class="utp-dashboard-banner">
        <span class="utp-university-name">UNIVERSITI TEKNOLOGI PETRONAS</span>
        <span class="utp-header-label">Academic Dashboard</span>
        </div>""",unsafe_allow_html=True)
    left,right=st.columns([8.6,1.4],vertical_alignment='center',gap='small')
    left.markdown(f'<div class="utp-page-title">{escape(title)}</div>',unsafe_allow_html=True)
    if right.button('⌂ Home',key=f'home_{title}',use_container_width=True):
        for k in ['role','principal','route','programme_filter','selected_student']:
            st.session_state.pop(k,None)
        st.rerun()
    st.caption(f'Last Updated: **{updated}** · Results reflect the latest academic records in the project database.')


def menu(role: str, principal: str) -> tuple[str,str]:
    if st.session_state.pop('go_individual', False):
        st.session_state.route = 'Individual Student'
    st.sidebar.markdown('### UTP Academic Dashboard')
    st.sidebar.caption('Admin' if role=='admin' else 'Academic Advisor' if role=='lecturer' else 'Student')
    route=st.sidebar.radio('Navigation Dashboard', ['Overall Cohort','Individual Student'],
                           key='route') if role!='student' else 'Individual Student'
    st.sidebar.divider()
    st.sidebar.markdown('**Filter by Programme:**')
    opts=['All courses','IT','CS','IS'] if role!='student' else []
    if opts:
        filter_opt=st.sidebar.radio('Programme',opts,
             format_func=lambda x:'All courses' if x=='All courses' else PROGRAMME_LABELS[x],key='programme_filter')
    else:
        filter_opt='All courses'
    st.sidebar.divider()
    if role=='lecturer':
        lecturer=db['directory'].loc[db['directory'].lecturer_code.eq(principal),'lecturer']
        if not lecturer.empty: st.sidebar.caption(f'Advisor: {lecturer.iloc[0]}')
    st.sidebar.caption('ML results support advisor review; they do not determine academic decisions.')
    if st.sidebar.button('Sign out'):
        for key in ['role','principal','route','programme_filter','selected_student']:
            st.session_state.pop(key,None)
        st.rerun()
    return route,filter_opt


def programme_scope(scope:pd.DataFrame, choice:str) -> pd.DataFrame:
    return scope if choice=='All courses' else scope[scope.programme_code.eq(choice)]


def kpis_overall(view: pd.DataFrame) -> None:
    probs=view[view.risk_probability.notna()]
    n_at=int(probs.predicted_at_risk.eq(True).sum())
    failures=int(view.failed_courses_all.sum())
    vals=[('Students',str(view.student_key.nunique())),('Average GPA',f'{view.gpa.mean():.2f}' if view.gpa.notna().any() else 'N/A'),
          ('Average CGPA',f'{view.cgpa.mean():.2f}' if view.cgpa.notna().any() else 'N/A'),
          ('Students Predicted At-Risk',str(n_at)),('At-Risk Rate',f'{n_at/len(probs):.1%}' if len(probs) else 'N/A'),
          ('Courses Failed',str(failures))]
    cols=st.columns(6,gap='small')
    for c,(label,value) in zip(cols,vals): c.metric(label,value)
    st.caption(f'ML predictions available for {len(probs)} of {len(view)} students in this view; prediction threshold: {bundle.get("threshold",RISK_LIMIT):.1%}' if bundle else 'Risk predictions unavailable: model file missing.')


def course_summary(view:pd.DataFrame) -> pd.DataFrame:
    rows=[]
    align=db['student_timeline_alignment']
    for p in ['IT','CS','IS']:
        sub=view[view.programme_code.eq(p)]
        if sub.empty: continue
        h=align[align.student_key.isin(sub.student_key)]
        q=h.grade.astype(str).str.upper().str.strip()
        covered=sub[sub.risk_probability.notna()]
        rows.append({'Programme':PROGRAMME_LABELS[p],'Code':p,'Students':len(sub),
                     'Average GPA':sub.gpa.mean(),'Average CGPA':sub.cgpa.mean(),
                     'Frequency of QF':int(q.eq('QF').sum()),'Frequency of F':int(q.eq('F').sum()),
                     'Predicted At-Risk':int(covered.predicted_at_risk.eq(True).sum()),
                     'Average Risk Probability (%)':float(covered.risk_probability.mean()*100) if len(covered) else np.nan,
                     'Predicted Coverage':len(covered)})
    return pd.DataFrame(rows)


def distribution_pie(view:pd.DataFrame) -> None:
    counts=view.programme_code.value_counts().reindex(['IT','CS','IS']).fillna(0)
    if not len(view): return empty_chart('Student Distribution')
    f=figure('Student Distribution',345)
    f.add_trace(go.Pie(labels=counts.index.tolist(),values=counts.tolist(),textinfo='label+percent',
                       textposition='inside',marker=dict(colors=[COLORS[p] for p in counts.index]),
                       hovertemplate='%{label}<br>%{value} students<br>%{percent}<extra></extra>'))
    show_chart(f)



def grade_distribution(view:pd.DataFrame, individual:bool=False) -> None:
    title='Grade Distribution'
    grade_rows=db['student_timeline_alignment'][db['student_timeline_alignment'].student_key.isin(view.student_key)]
    if grade_rows.empty:return empty_chart(title)
    grade_rows=grade_rows.copy()
    grade_rows['grade_clean']=grade_rows.grade.astype('string').str.upper().str.strip()
    grade_rows=grade_rows[grade_rows.grade_clean.notna() & grade_rows.grade_clean.ne('')]
    if grade_rows.empty:return empty_chart(title)
    counts=grade_rows.grade_clean.value_counts().sort_index()
    colours=['#1c779d','#23a990','#d5a43b','#ec7554','#8661a6','#e35e7f','#6b8b3c','#5f95cc']
    if individual:
        extras=[f'{int(counts[g])} course attempt(s), {int(counts[g])/len(grade_rows):.1%} of this student’s grades' for g in counts.index]
    else:
        extras=[]
        for g in counts.index:
            detail=[]
            for programme in ['IT','CS','IS']:
                prog=grade_rows[grade_rows.programme_code.eq(programme)]
                freq=int(prog.grade_clean.eq(g).sum())
                detail.append(f'{programme}: {freq} ({freq/len(prog):.1%} of {programme} attempts)' if len(prog) else f'{programme}: 0 (N/A)')
            extras.append('<br>'.join(detail))
    f=figure(title,345 if not individual else 265)
    f.add_trace(go.Pie(labels=counts.index.tolist(),values=counts.values,hole=.56,
         customdata=extras,marker=dict(colors=[colours[i%len(colours)] for i in range(len(counts))]),
         hovertemplate='%{label}: %{value} (%{percent})<br>%{customdata}<extra></extra>'))
    f.update_layout(legend=dict(orientation='h',x=.5,xanchor='center',y=-.20),
                    margin=dict(l=18,r=18,t=50,b=63))
    show_chart(f)

def performance_bars(summary:pd.DataFrame) -> None:
    if summary.empty:return empty_chart('Programme Performance (GPA/CGPA)')
    f=figure('Programme Performance (GPA/CGPA)',345)
    for k,color in [('Average GPA','#1085b3'),('Average CGPA','#50b9a7')]:
        f.add_bar(name=k,x=summary.Code,y=summary[k],marker_color=color,
                  hovertemplate='%{x}<br>'+k+': %{y:.2f}<extra></extra>')
    f.update_layout(barmode='group',yaxis=dict(range=[0,4],title='Grade point'))
    show_chart(f)


def risks_chart(summary:pd.DataFrame) -> None:
    if summary.empty:return empty_chart('At-Risk Predictions by Programme')
    pct=[100*r['Predicted At-Risk']/r['Predicted Coverage'] if r['Predicted Coverage'] else 0 for _,r in summary.iterrows()]
    f=figure('At-Risk Predictions by Programme',305)
    f.add_bar(x=summary.Code,y=pct,marker_color=[COLORS[p] for p in summary.Code],
              customdata=np.stack([summary['Predicted At-Risk'],summary['Predicted Coverage']],axis=1),
              hovertemplate='%{x}: %{y:.1f}%<br>At-risk %{customdata[0]} / %{customdata[1]} predicted<extra></extra>')
    f.update_layout(yaxis=dict(title='Predicted at-risk (%)',range=[0,100]))
    show_chart(f)


def standing_donut(view:pd.DataFrame) -> None:
    if view.empty:return empty_chart('Distribution of Academic Standing')
    s=view.standing.fillna('Not recorded').replace('', 'Not recorded')
    counts=s.value_counts()
    programme_breakdown=[]
    for standing in counts.index:
        group=view.loc[s.eq(standing),'programme_code'].value_counts()
        programme_breakdown.append(' / '.join(f'{p}: {int(group.get(p,0))}' for p in ['IT','CS','IS']))
    f=figure('Distribution of Academic Standing',305)
    f.add_trace(go.Pie(labels=counts.index,values=counts.values,hole=.56,
                       customdata=programme_breakdown,hovertemplate='%{label}: %{value} students (%{percent})<br>By programme: %{customdata}<extra></extra>'))
    f.update_layout(legend=dict(orientation='v',x=1,y=.95),margin=dict(l=5,r=5,t=52,b=8))
    show_chart(f)


def credit_bars(view:pd.DataFrame) -> None:
    f=figure('Credit Hour Status',305)
    data=[]
    for p in ['IT','CS','IS']:
        cohort=view[view.programme_code.eq(p)]
        if cohort.empty: continue
        credits=[credit_status(db['student_timeline_alignment'][db['student_timeline_alignment'].student_key.eq(key)],
                               db['curriculum_timeline'],p) for key in cohort.student_key]
        totals={n:sum(v[n] for v in credits) for n in ['passed','failed','remaining','required']}
        total=totals['required'] or 1
        data.append((p,100*totals['passed']/total,100*totals['failed']/total,100*totals['remaining']/total))
    if not data:return empty_chart('Credit Hour Status')
    for ind,label,color in [(1,'Passed credits',PASS_COLOR),(2,'Failed credits',FAIL_COLOR),(3,'Remaining credits',REMAIN_COLOR)]:
        f.add_bar(y=[d[0] for d in data],x=[d[ind] for d in data],name=label,orientation='h',
                  marker_color=color,hovertemplate='%{y}<br>'+label+': %{x:.1f}%<extra></extra>')
    f.update_layout(barmode='stack',xaxis=dict(range=[0,100],title='Curriculum credits (%)'),
                    yaxis=dict(autorange='reversed'),
                    height=350,margin=dict(l=38,r=20,t=55,b=100),
                    legend=dict(orientation='h',y=-.48,x=.5,xanchor='center'))
    show_chart(f)
    st.caption('Only passing grades count toward 122 (IT/IS) or 123 (CS) required credits. Failed credits require retaking; the yellow portion represents credits not yet attempted. Mapped records only.')


def summary_table(summary:pd.DataFrame) -> None:
    st.markdown('#### Programme Performance Overview')
    if summary.empty:st.info('No programme data.');return
    display=summary.drop(columns=['Code','Predicted Coverage']).copy()
    st.dataframe(display,use_container_width=True,hide_index=True,
                 column_config={'Average GPA':st.column_config.NumberColumn(format='%.2f'),
                                'Average CGPA':st.column_config.NumberColumn(format='%.2f'),
                                'Average Risk Probability (%)':st.column_config.NumberColumn(format='%.1f%%')})


def individual_roster_table(view:pd.DataFrame) -> None:
    st.markdown('#### Overall Individual Student Performance')
    if view.empty:st.info('No students in this filter.');return
    display=pd.DataFrame({
        'Student':view.student_name.fillna('—'),'Programme':view.programme_code,
        'Student Email':view.student_utp_email.fillna('Not provided'),
        'Academic Advisor':view.lecturer.fillna('Not assigned'),
        'Academic Advisor Email':view.lecturer_utp_email.fillna('Not provided'),
        'Current Semester':view.current_semester_label,
        'GPA':view.gpa,'CGPA':view.cgpa,
        'Standing':view.standing.fillna('Not recorded'),
        'Failed Courses (QF/F)':view.failed_courses_all,
        'At-Risk Probability (%)':view.risk_probability*100,
        'At-Risk Prediction':view.predicted_at_risk.fillna(False).astype(bool),
    })
    event=st.dataframe(display,use_container_width=True,hide_index=True,
                 on_select='rerun',selection_mode='single-row',key='admin_overall_student_table',
                 column_config={'GPA':st.column_config.NumberColumn(format='%.2f'),
                                'CGPA':st.column_config.NumberColumn(format='%.2f'),
                                'At-Risk Probability (%)':st.column_config.NumberColumn(format='%.1f%%'),
                                'At-Risk Prediction':st.column_config.CheckboxColumn('At-Risk Prediction')})
    selected=event.selection.rows if event is not None else []
    if selected:
        st.session_state.selected_student=str(view.iloc[selected[0]].student_key)
        st.session_state.go_individual=True
        st.rerun()


def advisor_overview(view:pd.DataFrame) -> None:
    st.markdown('#### Student Overview')
    if view.empty:st.info('No assigned students in the current programme filter.');return
    view=view.sort_values('student_name').reset_index(drop=True)
    credit_values=[]
    for key,programme in zip(view.student_key,view.programme_code):
        attempts=db['student_timeline_alignment'][db['student_timeline_alignment'].student_key.eq(key)]
        credit_values.append(credit_status(attempts,db['curriculum_timeline'],programme))
    overview=pd.DataFrame({'No.':np.arange(1,len(view)+1), 'Student Name':view.student_name,
        'Student ID':view.matric_number,'Student Email':view.student_utp_email.fillna('Not provided'),
        'Programme':view.programme_code,'Current Semester':view.current_semester_label,
        'GPA':view.gpa,'CGPA':view.cgpa,'Standing':view.standing.fillna('Not recorded'),
        'Credit Hours Taken':view.accumulated_credit_hours,
        'Credit Hours Passed':[x['passed'] for x in credit_values],
        'Credit Hours Failed':[x['failed'] for x in credit_values],
        'Courses Taken':view.courses_taken_all,'Failed Courses':view.failed_courses_all,
        'At-Risk Prediction (%)':view.risk_probability*100,
        'At-Risk':view.predicted_at_risk.fillna(False).astype(bool)})
    def risk_shade(value):
        if pd.isna(value):return ''
        return 'background-color:#d7f4e4' if value<12 else 'background-color:#fff1ad' if value<24 else 'background-color:#f8c8c8'
    styled=overview.style.map(risk_shade,subset=['At-Risk Prediction (%)'])
    event=st.dataframe(styled,use_container_width=True,hide_index=True,
        on_select='rerun',selection_mode='single-row',key='advisor_student_overview',
        column_config={'At-Risk Prediction (%)':st.column_config.NumberColumn('At-Risk Prediction (%)',format='%.1f%%'),
                       'At-Risk':st.column_config.CheckboxColumn('At-Risk'),
                       'GPA':st.column_config.NumberColumn(format='%.2f'),
                       'CGPA':st.column_config.NumberColumn(format='%.2f')})
    rows=event.selection.rows if event is not None else []
    if rows:
        st.session_state.selected_student=str(view.iloc[rows[0]].student_key)
        st.session_state.go_individual=True
        st.rerun()
    st.caption('Select one row to open that student’s dashboard. Only students assigned to this lecturer are visible.')


def render_overall(view:pd.DataFrame,role:str) -> None:
    heading('Academic Advisor – Overall Student Dashboard' if role=='lecturer' else 'Overall Dashboard')
    kpis_overall(view)
    if role=='lecturer': advisor_overview(view)
    a,b,c=st.columns(3,gap='medium')
    with a:distribution_pie(view)
    with b:grade_distribution(view)
    with c:performance_bars(course_summary(view))
    summary=course_summary(view)
    summary_table(summary)
    a,b,c=st.columns(3,gap='medium')
    with a:risks_chart(summary)
    with b:standing_donut(view)
    with c:credit_bars(view)
    if role=='lecturer':
        failed_courses_overall(view)
    else:
        individual_roster_table(view)



def failed_courses_overall(view:pd.DataFrame) -> None:
    st.markdown('#### Overall Failed Courses')
    a,b=st.columns([1.65,1],gap='medium')
    history=db['student_timeline_alignment']
    rows=history[history.student_key.isin(view.student_key)].copy()
    rows=rows[rows.grade.astype(str).str.upper().str.strip().isin(FAIL)].copy()
    with a:
        if rows.empty:st.info('No failed courses in the selected students.')
        else:
            # Count distinct students per course and programme, not repeat attempts.
            rows['Credit Hours']=pd.NA
            for programme in ['IT','CS','IS']:
                mask=rows.programme_code.eq(programme)
                rows.loc[mask,'Credit Hours']=attempts_credits(rows.loc[mask],programme)
            grouped=rows.groupby(['module_code_normalised','programme_code']).student_key.nunique().unstack(fill_value=0)
            courses=rows.sort_values('semester_seq').drop_duplicates('module_code_normalised')
            display=pd.DataFrame({'Course Code':courses.module_code.values,
                 'Course Name':courses.module.values,'Credit Hours':courses['Credit Hours'].values,
                 'Course Group':courses.course_group.values})
            for prog in ['IT','CS','IS']:
                display[prog]=courses.module_code_normalised.map(grouped[prog]).fillna(0).astype(int).values if prog in grouped.columns else 0
            st.dataframe(display,use_container_width=True,hide_index=True,height=360)
    with b:
        f=figure('F and QF by Programme',355)
        for grade,color in [('F','#d95056'),('QF','#efb856')]:
            counts=rows[rows.grade.astype(str).str.upper().str.strip().eq(grade)].programme_code.value_counts()
            f.add_bar(name=grade,x=['IT','CS','IS'],y=[int(counts.get(p,0)) for p in ['IT','CS','IS']],marker_color=color)
        f.update_layout(barmode='group',yaxis_title='Failed course attempts',
                 margin=dict(l=45,r=15,t=60,b=75),legend=dict(orientation='h',y=-.26,x=.5,xanchor='center'))
        show_chart(f)

def select_student(view:pd.DataFrame,role:str) -> str | None:
    if view.empty:return None
    keys=view.student_key.astype(str).tolist()
    previous=st.session_state.get('selected_student')
    if previous not in keys: st.session_state.selected_student=keys[0]
    if role=='student':return st.session_state.selected_student
    names=dict(zip(view.student_key,view.student_name))
    ids=dict(zip(view.student_key,view.matric_number.astype(str)))
    # The selectboxes independently search by name/ID and update the same canonical key.
    def choose_name():st.session_state.selected_student=st.session_state['pick_name']
    def choose_id():st.session_state.selected_student=st.session_state['pick_id']
    def choose_advisor():
        code=st.session_state.get('pick_advisor')
        if code!='All advisors':
            hits=view.loc[view.lecturer.fillna('Not assigned').eq(code)]
            if not hits.empty:st.session_state.selected_student=str(hits.iloc[0].student_key)
    current=st.session_state.selected_student
    if role=='admin':
        a,b,c=st.columns(3)
    else:
        a,b=st.columns(2)
    st.session_state.pick_name=current
    st.session_state.pick_id=current
    with a:
        st.selectbox('Student Name',keys,format_func=lambda key:names.get(key,key),key='pick_name',on_change=choose_name)
    with b:
        st.selectbox('Student ID',keys,format_func=lambda key:ids.get(key,key),key='pick_id',on_change=choose_id)
    if role=='admin':
        options=['All advisors',*sorted(view.lecturer.dropna().astype(str).unique().tolist())]
        with c:
            st.selectbox('Academic Advisor',options,key='pick_advisor',on_change=choose_advisor)
    return st.session_state.selected_student


def risk_indicator(prob:object) -> None:
    st.markdown('#### Risk Probability')
    if pd.isna(prob):st.info('ML prediction unavailable for this student.');return
    pct=float(prob)*100
    color=PASS_COLOR if pct<10 else '#e29b35' if pct<20 else FAIL_COLOR
    f=figure('',245)
    f.add_bar(y=['Risk'],x=[pct],orientation='h',marker_color=color,width=.43,
              hovertemplate='Predicted risk: %{x:.1f}%<extra></extra>')
    f.update_layout(xaxis=dict(range=[0,100],title='Risk probability (%)',ticksuffix='%'),
                    yaxis=dict(showticklabels=False),showlegend=False,
                    margin=dict(l=25,r=25,t=24,b=55))
    f.add_vline(x=24,line_dash='dash',line_color='#91acb9',annotation_text='Decision threshold: 24%',
                annotation_position='top right')
    show_chart(f)
    st.caption('High risk if at or above the trained model threshold (approximately 24%).')


def student_credit_chart(course_attempts:pd.DataFrame,programme:str) -> dict:
    x=credit_status(course_attempts,db['curriculum_timeline'],programme)
    f=figure('Completed Credit Hours',265)
    vals=[x['passed'],x['failed'],x['remaining']]
    f.add_trace(go.Pie(labels=['Passed','Failed (F/QF)','Remaining'],values=vals,hole=.6,
                       marker=dict(colors=[PASS_COLOR,FAIL_COLOR,REMAIN_COLOR]),
                       textinfo='label+percent',hovertemplate='%{label}: %{value:.0f} credits<extra></extra>'))
    f.update_layout(margin=dict(l=12,r=12,t=45,b=68),legend=dict(orientation='h',y=-.25,x=.5,xanchor='center'))
    show_chart(f)
    st.caption(f"{x['passed']:.0f} passing credits toward {x['required']} required. {x['failed']:.0f} failed credits require retaking; {x['remaining']:.0f} credits not yet attempted. {x['credits_to_pass']:.0f} credits still need passing grades. {x['unmapped']} attempts are unmapped.")
    return x


def trend_chart(history:pd.DataFrame) -> None:
    st.markdown('#### GPA/CGPA Trend and Forecast')
    history=history[history[['gpa','cgpa']].notna().any(axis=1)]
    if history.empty:return st.info('No GPA/CGPA history recorded.')
    h=forecast_academic_trend(history)
    last=int(history.semester_seq.max())
    f=figure('Historical Performance and Ridge Regression Forecast',355)
    for col,label,color in [('Historical GPA','GPA (recorded)','#117db7'),
                            ('Historical CGPA','CGPA (recorded)','#219b72'),
                            ('Forecast GPA','GPA (forecast)','#bc64d7'),
                            ('Forecast CGPA','CGPA (forecast)','#e0a648')]:
        points=h['Semester'].mask(h['Semester'].eq(last),last+.5) if 'forecast' in label else h['Semester']
        f.add_trace(go.Scatter(x=points,y=h[col],mode='lines+markers',name=label,
                               line=dict(color=color,dash='dot' if 'forecast' in label else 'solid',width=3),
                               connectgaps=False))
    f.add_vline(x=last+.5,line_dash='dash',line_color='#607788',
                annotation_text='Recorded  |  Forecast',annotation_position='top left')
    f.add_hline(y=2,line_dash='dot',line_color='#a6b3be',annotation_text='GPA 2.00 reference')
    f.update_layout(xaxis=dict(title='Semester',dtick=1),yaxis=dict(title='Grade point',range=[0,4.1]),
                    legend=dict(orientation='h',y=-.35),margin=dict(l=43,r=35,t=54,b=80))
    show_chart(f)
    st.caption('Forecast uses per-student regularised (Ridge) regression, trained on recorded grades only. Exploratory projections are not guaranteed grades or the academic-risk classifier.')


def semester_controls(timeline:pd.DataFrame, student_key:str, industrial:dict[str,int]) -> int:
    st.markdown('#### Select Semester — Academic Timeline')
    st.caption('Current semester highlighted · select a semester to view its subjects. 🔒 = SIT/SIP (actual placement overrides curriculum default).')
    current_rows=timeline[timeline.Status.eq('Current')]
    now=int(current_rows.iloc[0]['Semester Number']) if not current_rows.empty else 1
    valid=set(timeline['Semester Number'].astype(int))
    state_key=f'semester_choice_{student_key}'
    if st.session_state.get(state_key) not in valid:st.session_state[state_key]=now
    selected=int(st.session_state[state_key])
    for start in range(1,len(timeline)+1,5):
        cols=st.columns(5,gap='small')
        for c,n in zip(cols,range(start,min(start+5,len(timeline)+1))):
            label=f'Semester {n}'
            if n==now: label+=' 📍'
            if n in industrial.values():label+=' 🔒'
            with c:
                if st.button(label,key=f'semester_{student_key}_{n}',use_container_width=True,
                             type='primary' if n==selected else 'secondary'):
                    st.session_state[state_key]=n
                    st.rerun()
    return selected



def attempts_credits(frame:pd.DataFrame, programme:str) -> pd.Series:
    return course_credit_hours(frame,db['curriculum_timeline'],programme,
                               db.get('curriculum_options'))


def course_table_order(frame:pd.DataFrame) -> pd.DataFrame:
    if 'Course Code' in frame.columns and 'Course Name' in frame.columns and 'Credit Hours' in frame.columns:
        cols=list(frame.columns)
        cols.remove('Credit Hours')
        cols.insert(cols.index('Course Name')+1,'Credit Hours')
        return frame[cols]
    return frame

def render_subjects(student_key:str, chosen:int, programme:str,
                    history:pd.DataFrame, enrolled:pd.DataFrame) -> None:
    st.markdown(f'#### Subjects by Semester — Semester {chosen}')
    subject=history[history.semester_seq.eq(chosen)]
    current=enrolled[enrolled.semester_seq.eq(chosen)]
    curriculum=db['curriculum_timeline']
    defaults=curriculum[curriculum.programme_code.eq(programme)&curriculum.default_semester.eq(chosen)]
    if not subject.empty:
        display=pd.DataFrame({'Course Code':subject.module_code,'Course Name':subject.module,
           'Credit Hours':attempts_credits(subject,programme),'Course Group':subject.course_group.fillna('—'),'Grade':subject.grade.fillna('—'),
           'Default Semester':subject.default_semester,'Timeline Status':subject.timeline_status,
           'Pass/Fail':np.where(subject.grade.astype(str).str.upper().str.strip().isin(FAIL),'Fail','Pass')})
        st.dataframe(display,hide_index=True,use_container_width=True)
    elif not current.empty:
        display=pd.DataFrame({'Course Code':current.module_code,'Course Name':current.module,
          'Credit Hours':attempts_credits(current,programme),'Course Group':current.course_group.fillna('—'),'Grade':'Not released',
          'Default Semester':current.default_semester,'Timeline Status':current.timeline_status,
          'Pass/Fail':'Pending'})
        st.dataframe(display,hide_index=True,use_container_width=True)
    elif not defaults.empty:
        display=pd.DataFrame({'Course Code':defaults.course_code_raw,'Course Name':defaults.course_name,
            'Credit Hours':defaults.credit_hours,'Course Group':defaults.course_group,'Status':'Unrecorded',
            'Prerequisite':defaults.prerequisite_text.fillna('None')})
        st.dataframe(display,hide_index=True,use_container_width=True)
    else:st.info('No recorded/enrolled or default subjects for this semester.')
    with st.expander('Show default curriculum subjects for this semester'):
        if defaults.empty:st.info('No default curriculum entries for this semester.')
        else:st.dataframe(defaults[['course_code_raw','course_name','credit_hours','course_group','prerequisite_text']]
           .rename(columns={'course_code_raw':'Course Code','course_name':'Course Name','credit_hours':'Credit Hours',
                            'course_group':'Course Group','prerequisite_text':'Prerequisite'}),use_container_width=True,hide_index=True)
    actual=industrial_record(history,enrolled)
    industrial_here=[k for k,s in actual.items() if s==chosen]
    if industrial_here:
        st.info(f"Semester {chosen} is an actual {' / '.join(industrial_here)} semester. No additional courses are permitted in the pathway for this semester.")


def failed_and_incomplete(student_key:str,programme:str,course_attempts:pd.DataFrame,
                          probability:float|None) -> None:
    one,two=st.columns(2,gap='medium')
    failed=course_attempts[course_attempts.grade.astype(str).str.upper().str.strip().isin(FAIL)]
    ml=db['ml_semester']; semester_dates=ml[ml.student_key.eq(student_key)].drop_duplicates('semester_seq').set_index('semester_seq')
    with one:
        st.markdown('#### Failed Courses')
        if failed.empty:st.success('No F/QF attempts recorded.')
        else:
            fmt=[]
            for r in failed.itertuples():
                term=semester_dates.loc[r.semester_seq] if r.semester_seq in semester_dates.index else None
                fmt.append({'Semester':semester_label(r.semester_seq,term.period if term is not None else '',term.year if term is not None else np.nan),
                            'Course Code':r.module_code,'Course Name':r.module,'Credit Hours':attempts_credits(failed,programme).loc[r.Index],'Grade':r.grade,
                            'Timeline Status':r.timeline_status,'Action':course_action(r.grade,probability)})
            st.dataframe(pd.DataFrame(fmt),use_container_width=True,hide_index=True)
    with two:
        st.markdown('#### Repeat / Incomplete Courses')
        # Failed courses passed on a later attempt must not be recommended again.
        taken=course_attempts.sort_values('semester_seq').copy()
        taken['course_key']=taken.curriculum_row_id.fillna(taken.module_code_normalised)
        latest=taken.drop_duplicates('course_key',keep='last')
        needs=latest[latest.grade.astype(str).str.upper().str.strip().isin(FAIL)]
        rows=[{'Course Code':r.module_code,'Course Name':r.module,'Credit Hours':attempts_credits(needs,programme).loc[r.Index],'Status':'Failed',
               'Action':course_action(r.grade,probability)} for r in needs.itertuples()]
        curriculum=db['curriculum_timeline'][db['curriculum_timeline'].programme_code.eq(programme)]
        handled=set(taken.curriculum_row_id.dropna())
        for r in curriculum.itertuples():
            if r.curriculum_row_id in handled or bool(r.is_selection_slot):continue
            rows.append({'Course Code':r.course_code_raw,'Course Name':r.course_name,'Credit Hours':r.credit_hours,'Status':'Unrecorded',
                         'Action':'Verify eligibility and degree requirement with advisor'})
        if not rows:st.success('No identified repeat/incomplete curriculum requirements.')
        else:st.dataframe(pd.DataFrame(rows).drop_duplicates(['Course Code','Status']),
                          hide_index=True,use_container_width=True,height=295)
    st.caption('Suggested actions are rules informed by ML risk and course history; they are not separate trained recommendation-model predictions.')


def alignment_panel(course_attempts:pd.DataFrame,ml:pd.DataFrame,programme:str) -> None:
    st.markdown('#### Full Course-to-Curriculum Timeline Alignment')
    a,b=st.columns([2,3],gap='medium')
    with a:
        counts=course_attempts.timeline_status.value_counts().reindex(
            ['On default semester','Repeat attempt','Taken later','Unmapped to current curriculum'],fill_value=0)
        f=figure('Timeline Status',300)
        f.add_bar(x=counts.index.tolist(),y=counts.values.tolist(),marker_color=['#1f95b9','#efb856','#a278ba','#d95b61'])
        f.update_layout(xaxis=dict(tickangle=-23),yaxis=dict(title='Course attempts'),margin=dict(l=25,r=5,t=52,b=93))
        show_chart(f)
        earlier=int(course_attempts.timeline_status.eq('Taken earlier').sum())
        if earlier:st.caption(f'{earlier} additional course attempts are classified as “Taken earlier” in the full table.')
    with b:
        if course_attempts.empty:st.info('No course attempts recorded.')
        else:
            lookup=ml.drop_duplicates('semester_seq').set_index('semester_seq')
            rows=[]
            for r in course_attempts.itertuples():
                term=lookup.loc[r.semester_seq] if r.semester_seq in lookup.index else None
                rows.append({'Semester':semester_label(r.semester_seq,term.period if term is not None else '',term.year if term is not None else np.nan),
                             'Course Code':r.module_code,'Course Name':r.module,
                             'Credit Hours':attempts_credits(course_attempts,programme).loc[r.Index],
                             'Course Group':r.course_group,'Grade':r.grade,'Default Semester':r.default_semester,'Timeline Status':r.timeline_status})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,height=310)


def render_pathway(student_key:str,programme:str, ml:pd.DataFrame,course_attempts:pd.DataFrame,
                   enrolled:pd.DataFrame,credits:dict) -> None:
    st.markdown('#### Generate Next Semester Pathway')
    if ml.empty:return st.info('Pathway generation requires semester academic history.')
    latest=ml.sort_values('semester_seq').iloc[-1]
    if credits['complete']:
        st.success(f"Curriculum credit target of {credits['required']} mapped passed credits satisfied. Advisor should verify formal graduation requirements.")
        return
    active_semester=max(int(ml.semester_seq.max()),int(enrolled.semester_seq.max()) if not enrolled.empty else 0)
    next_sem=active_semester+1
    fixed=planned_industrial_course(db['curriculum_timeline'],programme,next_sem,course_attempts,enrolled)
    if not fixed.empty:
        limit=int(fixed.credit_hours.sum())
        st.info(f"Semester {next_sem} is reserved for {fixed.iloc[0].course_name} alone ({limit} credits). SIT/SIP placement follows the student's actual history.")
    else:
        plan=db['semester_credit_plan']
        planned=plan[plan.programme_code.eq(programme)&plan.default_semester.eq(next_sem)]
        default=int(planned.iloc[0].planned_credit_hours) if len(planned) else 15
        limit=st.number_input('Maximum credit hours',min_value=1,max_value=24,value=min(default,24),key=f'limit_{student_key}')
    if st.button('Generate Next Semester',type='primary',key=f'generate_{student_key}'):
        with st.spinner('Checking programme prerequisites and generating pathway…'):
            try:
                result=generate_pathway(student_key,int(limit),DATA)
                st.session_state[f'pathway_{student_key}']=result
                st.session_state.pop(f'future_pathway_{student_key}',None)
                st.session_state.pop(f'future_limit_used_{student_key}',None)
            except Exception as e:st.error(f'Pathway could not be generated: {e}');return
    result=st.session_state.get(f'pathway_{student_key}')
    if result is None:return
    reference=latest
    if not enrolled.empty and int(enrolled.semester_seq.max())>int(latest.semester_seq):
        reference=enrolled.sort_values('semester_seq').iloc[-1]
    nxt_year,nxt_period=next_period(int(reference.year),str(reference.period))
    selected=result['recommendations'].copy()
    st.markdown('#### Recommended Next Semester Courses')
    if selected.empty:st.info('No eligible named courses were selected automatically.')
    else:
        selected['Semester Number']=next_sem
        selected['Semester']=nxt_period
        selected['Year']=nxt_year
        selected=selected.rename(columns={'course_code':'Course Code','course_name':'Course Name',
                                          'course_group':'Course Group','credit_hours':'Credit Hours',
                                          'eligible':'Eligible','prerequisite_status':'Prerequisite',
                                          'priority_score':'Priority Score','reason':'Reason'})
        cols=['Semester Number','Semester','Year','Course Code','Course Name','Credit Hours','Course Group',
              'Eligible','Prerequisite','Priority Score','Reason']
        st.dataframe(selected[cols],use_container_width=True,hide_index=True,
                     column_config={'Eligible':st.column_config.CheckboxColumn('Eligible')})
        if result['summary']['fixed_industrial_semester'] and not selected['Eligible'].all():
            st.warning('Reserved industrial course is provisional: prerequisite eligibility is not confirmed. Advisor verification is required before enrollment.')
    if not result['selection_requirements'].empty:
        with st.expander('Curriculum course selections required'):
            st.dataframe(result['selection_requirements'].rename(columns={'requirement':'Requirement',
                            'credit_hours':'Credit Hours','reason':'Reason'}),use_container_width=True,hide_index=True)
    omit={'curriculum_row_id','candidate_kind','eligible_option_count','source_sheet'}
    for section,label in [('held_for_manual_review','Held for prerequisite or legacy-equivalency review'),
                          ('deferred_due_requirements','Due requirements deferred by the selected credit limit')]:
        frame=result[section]
        if not frame.empty:
            with st.expander(label):
                formatted=frame.drop(columns=[c for c in omit if c in frame.columns]).rename(columns={
                    'course_code':'Course Code','course_name':'Course Name','credit_hours':'Credit Hours'})
                st.dataframe(course_table_order(formatted),use_container_width=True,hide_index=True)
    if not result['unmapped_failed_courses'].empty:
        with st.expander('Unmapped failed courses requiring advisor review'):
            raw_unmapped=result['unmapped_failed_courses']
            failed_unmapped=raw_unmapped.rename(columns={'module_code':'Course Code','module':'Course Name',
                'grade':'Grade','semester_seq':'Semester'})
            failed_unmapped.insert(failed_unmapped.columns.get_loc('Course Name')+1,'Credit Hours',attempts_credits(raw_unmapped,programme))
            st.dataframe(failed_unmapped,use_container_width=True,hide_index=True)
    st.markdown('#### Recommended Future Semester Courses')
    future_limit=int(st.number_input('Maximum credit hours for future semesters', min_value=1,
                                     max_value=24, value=16, step=1, key=f'future_limit_{student_key}'))
    # This limit is independent of the recommended *next* semester's limit.
    # Recalculate when it changes, so the +/- input has an immediate effect.
    if st.session_state.get(f'future_limit_used_{student_key}') != future_limit or \
            f'future_pathway_{student_key}' not in st.session_state:
        with st.spinner('Updating future-semester recommendations…'):
            try:
                st.session_state[f'future_pathway_{student_key}']=generate_future_pathway(
                    student_key,int(limit),DATA,initial=result,future_maximum_credits=future_limit)
                st.session_state[f'future_limit_used_{student_key}']=future_limit
            except Exception as e:
                st.error(f'Future-semester pathway could not be generated: {e}')
                return
    future=st.session_state.get(f'future_pathway_{student_key}')
    if future is None:
        st.caption('Generate the next semester pathway to produce a provisional plan for later semesters.')
    elif future['courses'].empty:
        st.info('No additional future courses could be allocated automatically. Outstanding requirements need advisor review.' if future['incomplete'] else 'No additional future-semester courses required.')
    else:
        table=future['courses']
        sems=sorted(int(n) for n in table['Semester Number'].unique())
        choice_key=f'future_semester_choice_{student_key}'
        if st.session_state.get(choice_key) not in sems:st.session_state[choice_key]=sems[0]
        st.selectbox('Select future semester',sems,format_func=lambda n:f'Semester {n}',key=choice_key)
        visible=table[table['Semester Number'].eq(st.session_state[choice_key])]
        st.dataframe(visible,use_container_width=True,hide_index=True,
                     column_config={'Eligible':st.column_config.CheckboxColumn('Eligible')})
        st.caption(f"Provisional plan through Semester {future['last_semester']}. Planned passing credits: {future['passed_credits_planned']:.0f} / {future['required_credits']} required. All planned course passes and prerequisites require confirmation.")
    if future is not None and future['incomplete']:
        st.warning('The provisional plan has outstanding credits/requirements that cannot be placed automatically; academic-advisor review is required.')
    st.caption('Recommendations are prerequisite-checked optimization outputs, using ML risk for advising context. Advisor approval is required.')


def render_individual(view:pd.DataFrame,role:str,principal:str) -> None:
    heading('Academic Advisor – Individual Student Dashboard' if role=='lecturer' else 'Individual Student Dashboard')
    if view.empty:return st.warning('No accessible students for this programme.')
    student_key=select_student(view,role)
    if not student_key or student_key not in set(view.student_key):return st.error('Access not permitted.')
    info=view[view.student_key.eq(student_key)].iloc[0]
    ml=db['ml_semester'][db['ml_semester'].student_key.eq(student_key)].sort_values('semester_seq')
    attempt=db['student_timeline_alignment'][db['student_timeline_alignment'].student_key.eq(student_key)].sort_values(['semester_seq','module_code'])
    enrollment=db['current_enrollment_alignment'][db['current_enrollment_alignment'].student_key.eq(student_key)]
    if role=='student':
        a,b,c=st.columns(3)
        a.text_input('Student Name',value=str(info.student_name),disabled=True)
        b.text_input('Student ID',value=str(info.matric_number),disabled=True)
        c.text_input('Academic Advisor',value=str(info.lecturer) if pd.notna(info.lecturer) else 'Not assigned',disabled=True)
    metrics=[('Programme',info.programme_code),('Current Semester',info.current_semester_label),
             ('GPA',f'{info.gpa:.2f}' if pd.notna(info.gpa) else 'N/A'),
             ('CGPA',f'{info.cgpa:.2f}' if pd.notna(info.cgpa) else 'N/A'),
             ('Standing',str(info.standing) if pd.notna(info.standing) else 'N/A'),
             ('At-Risk Percentage',f'{info.risk_probability:.1%}' if pd.notna(info.risk_probability) else 'N/A')]
    cols=st.columns(6,gap='small')
    for c,(lab,value) in zip(cols,metrics): c.metric(lab,value)
    trend_chart(ml)
    a,b,c=st.columns([1.15,1,1],gap='medium')
    with a:risk_indicator(info.risk_probability)
    with b:grade_distribution(view[view.student_key.eq(student_key)],individual=True)
    with c:credits=student_credit_chart(attempt,info.programme_code)
    timeline=student_timeline(ml,enrollment,attempt)
    semester_credits = attempts_credits(attempt,info.programme_code).groupby(attempt['semester_seq']).sum()
    timeline.insert(timeline.columns.get_loc('Status')+1,'Credit Hours',
                    timeline['Semester Number'].map(semester_credits).fillna(0).astype(int))
    st.markdown('#### Semester Timeline Record')
    def mark_current(row):
        # Explicit high-contrast foreground also works in dark mode.
        return ['background-color:#167b83;color:#ffffff;font-weight:650'
                if row['Status']=='Current' else '' for _ in row]
    timeline['Year']=pd.to_numeric(timeline['Year'],errors='coerce').astype('Int64')
    st.dataframe(timeline.style.apply(mark_current,axis=1).format({'Year':'{:.0f}','GPA':'{:.2f}','CGPA':'{:.2f}'},na_rep='—'),
                 use_container_width=True,hide_index=True,height=min(520,65+35*len(timeline)))
    if credits['complete']:
        st.success(f"Programme requirement of {credits['required']} mapped, passed credit hours has been satisfied. Graduation still needs university verification.")
    elif credits['credits_to_pass']>0:
        st.info(f"{credits['credits_to_pass']:.0f} more credit hours must receive passing grades to meet the {credits['required']}-credit programme target. Failed courses do not count toward completion.")
    for alert in validate_industrial_history(attempt):st.warning('Data-quality check: '+alert)
    industrial=industrial_record(attempt,enrollment)
    chosen=semester_controls(timeline,student_key,industrial)
    render_subjects(student_key,chosen,info.programme_code,attempt,enrollment)
    failed_and_incomplete(student_key,info.programme_code,attempt,info.risk_probability)
    alignment_panel(attempt,ml,info.programme_code)
    render_pathway(student_key,info.programme_code,ml,attempt,enrollment,credits)


def render_dashboard_chat(role:str, principal:str, route:str, programme_filter:str,
                          view:pd.DataFrame, student_key:str|None) -> None:
    """Show curated, answerable prompts for both AI and offline modes."""
    st.divider()
    st.markdown('#### Academic Dashboard Chatbot')
    st.caption('Choose a suggested question or type your own. Answers use only data you can access.')
    chat_scope=f'{role}_{principal}_{route}_{programme_filter}_{student_key or "cohort"}'
    mode_key=f'dashboard_chat_ai_{role}_{principal}'
    use_ai=st.toggle('Generative AI',value=False,key=mode_key,
                     help='Off: answers computed locally from your dashboard. On: optional API-generated explanation, if configured.')
    mode='generative' if use_ai else 'offline'
    history_key=f'dashboard_chat_history_{chat_scope}_{mode}'
    api_key=os.environ.get('OPENAI_API_KEY','').strip()
    if use_ai and api_key:
        st.caption('Generative mode: questions and de-identified, access-limited academic data are sent to the configured AI API.')
    elif use_ai:
        st.info('No OPENAI_API_KEY is configured. Suggested questions still work using local, data-based answers.')
    else:
        st.caption('Local mode: answers are calculated from the dashboard. No external AI request is made.')

    if history_key not in st.session_state:
        st.session_state[history_key]=[]
    messages=st.session_state[history_key]
    suggestions=suggested_prompts(individual=bool(student_key),generative=use_ai)
    with st.container(border=True):
        select_col,send_col=st.columns([5,1.25],gap='small',vertical_alignment='bottom')
        with select_col:
            chosen_prompt=st.selectbox('Suggested questions', ['Select a question to ask…',*suggestions],
                key=f'dashboard_chat_suggestion_{chat_scope}_{mode}',
                help='Each question targets a different aspect of your academic dashboard.')
        with send_col:
            send_suggestion=st.button('Ask question',type='primary',use_container_width=True,
                disabled=chosen_prompt not in suggestions,
                key=f'dashboard_chat_ask_{chat_scope}_{mode}')
        st.divider()
        for msg in messages:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
        typed_prompt=st.chat_input('Or ask your own academic question…',
                                   key=f'dashboard_chat_input_{chat_scope}_{mode}')
        prompt=typed_prompt if typed_prompt else (chosen_prompt if send_suggestion else None)
        if prompt:
            prompt=prompt.strip()[:1500]
            if not prompt:return
            with st.chat_message('user'):
                st.markdown(prompt)
            context=build_dashboard_context(view,db['ml_semester'],db['student_timeline_alignment'],
                db['curriculum_timeline'],db['curriculum_options'],student_key=student_key,
                dashboard=route,programme_filter=programme_filter)
            if student_key:
                future=st.session_state.get(f'future_pathway_{student_key}')
                if future:
                    context['future_pathway']={
                        'planned_passing_credits':float(future['passed_credits_planned']),
                        'required_credits':int(future['required_credits']),
                        'incomplete':bool(future['incomplete']),
                        'semesters':sorted(int(n) for n in future['courses'].get('Semester Number',pd.Series(dtype=int)).unique())}
            if use_ai and api_key:
                try:
                    reply=generate_ai_answer(prompt,context,messages,api_key,
                        model=os.environ.get('OPENAI_MODEL','gpt-4o-mini'))
                except Exception as exc:
                    reply=(f'Generative AI is unavailable ({type(exc).__name__}). '
                           'Here is the dashboard-data answer instead:\n\n'
                           +answer_dashboard_question(prompt,context))
            else:
                reply=answer_dashboard_question(prompt,context)
            with st.chat_message('assistant'):
                st.markdown(reply)
            messages.extend([{'role':'user','content':prompt},{'role':'assistant','content':reply}])


@st.dialog('Dashboard Access')
def login_popup(role:str) -> None:
    titles={'admin':'Admin','student':'Student','lecturer':'Lecturer'}
    st.write(f'Enter your {titles[role]} access code.')
    with st.form('access_form'):
        code=st.text_input('Access code',type='password',placeholder='Enter access code')
        submit=st.form_submit_button('Enter Dashboard',use_container_width=True,type='primary')
    if submit:
        identity=auth_from_code(db['directory'],role,code.strip())
        if identity is None:st.error('Invalid access code. Please try again.')
        else:
            st.session_state.role=role
            st.session_state.principal=identity
            st.session_state.pop('route',None)
            st.session_state.pop('programme_filter',None)
            st.session_state.pop('selected_student',None)
            st.rerun()


def homepage() -> None:
    """Campus-photo landing screen with a true full-viewport university header."""
    image_data=base64.b64encode(CAMPUS_IMAGE.read_bytes()).decode('ascii')
    st.markdown(f"""<style>
      section[data-testid="stSidebar"], [data-testid="collapsedControl"] {{display:none!important;}}
      [data-testid="stAppViewContainer"] {{
        min-height:100vh;
        background:linear-gradient(115deg,rgba(4,23,40,.78),rgba(9,50,71,.50) 58%,rgba(4,26,40,.67)),
                   url('data:image/jpeg;base64,{image_data}') center center / cover fixed !important;
      }}
      [data-testid="stMainBlockContainer"] {{max-width:none!important;}}
      .block-container {{max-width:none!important;padding-top:0!important;padding-bottom:2rem!important;}}
      .utp-home-header {{
        position:relative;width:100vw;box-sizing:border-box;margin-left:calc(50% - 50vw);
        background:linear-gradient(95deg,rgba(3,29,50,.98),rgba(10,78,104,.97));
        border-bottom:3px solid #45cfce;color:#fff;text-align:center;
        padding:25px 20px;box-shadow:0 5px 25px rgba(0,0,0,.3);
      }}
      .utp-home-header span {{display:block;font-weight:800;letter-spacing:.09em;
        line-height:1.35;font-size:clamp(1.18rem,2.4vw,2.1rem);color:#fff;}}
      .utp-home-hero {{
        max-width:850px;margin:clamp(46px,9vh,115px) auto 26px;
        padding:clamp(28px,4vw,56px) 26px 34px;text-align:center;
        background:rgba(4,28,45,.78);backdrop-filter:blur(7px);
        border:1px solid rgba(204,247,247,.28);border-radius:22px;
        box-shadow:0 17px 46px rgba(0,0,0,.25);
      }}
      .utp-home-hero h1 {{color:#fff!important;font-size:clamp(2.15rem,5vw,4.3rem);
        font-weight:820;line-height:1.15;margin:0 0 15px;letter-spacing:-.02em;}}
      .utp-home-hero p {{color:#e0f5f5!important;font-size:clamp(.95rem,1.6vw,1.18rem);
        line-height:1.65;margin:0 auto;max-width:650px;}}
      .utp-entrance-intro {{text-align:center;color:#fff;font-size:1.05rem;
        font-weight:650;text-shadow:0 2px 4px rgba(0,0,0,.6);margin-bottom:12px;}}
      .utp-home-note {{color:#e9f5f8!important;text-align:center;
        text-shadow:0 2px 5px rgba(0,0,0,.8);margin-top:35px;}}
      .stButton button[kind="primary"] {{min-height:3.35rem;background:#097e91;
        border:1px solid #81d8df;box-shadow:0 6px 20px rgba(0,0,0,.28);
        font-size:1.05rem;color:#fff;}}
      .stButton button[kind="secondary"] {{border:1px solid #d1e8ec;background:rgba(7,31,51,.88);
        color:#fff;min-height:2.8rem;}}
      .stButton button[kind="secondary"]:hover {{color:#fff;background:#125f78;}}
      @media(max-width:600px) {{
         .utp-home-header {{padding:18px 12px;}}
         .utp-home-hero {{margin-top:48px;padding:28px 17px;}}
      }}
    </style>""",unsafe_allow_html=True)
    st.markdown('<div class="utp-home-header"><span>UNIVERSITI TEKNOLOGI PETRONAS</span></div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="utp-home-hero">
        <h1>Academic Dashboard</h1>
        <p>Explore academic progress, programme performance, early-warning insights,
        and personalised learning pathways in one place.</p>
        </div>""",unsafe_allow_html=True)
    st.markdown('<p class="utp-entrance-intro">Choose your dashboard</p>',unsafe_allow_html=True)
    _,student_col,lecturer_col,_=st.columns([1.1,1.3,1.3,1.1],gap='large')
    with student_col:
        if st.button('🎓 Student',use_container_width=True,type='primary'):
            login_popup('student')
    with lecturer_col:
        if st.button('👩‍🏫 Lecturer',use_container_width=True,type='primary'):
            login_popup('lecturer')
    st.write('')
    _,admin_col,_=st.columns([3,1.3,3])
    with admin_col:
        if st.button('Admin access',use_container_width=True,type='secondary'):
            login_popup('admin')
    st.markdown('<p class="utp-home-note">Demo access codes are provided in the project database. '
                'This prototype does not use institutional single sign-on.</p>',unsafe_allow_html=True)


role=st.session_state.get('role')
principal=st.session_state.get('principal')
if not role or not principal:
    homepage()
else:
    allowed=authorized_keys(db['directory'],role,principal)
    if not allowed:
        st.error('No students are assigned to this access code.')
        if st.button('Return Home'):
            st.session_state.clear()
            st.rerun()
        st.stop()
    scope=roster[roster.student_key.isin(allowed)].copy()
    route,choice=menu(role,principal)
    visible=programme_scope(scope,choice)
    if route=='Overall Cohort' and role!='student':render_overall(visible,role)
    else:render_individual(visible,role,principal)
    active_key=(st.session_state.get('selected_student') if route=='Individual Student' else None)
    if role=='student':
        active_key=next(iter(allowed))
    # Do not accidentally expose data from another student when a filter leaves
    # the individual dashboard empty or when changing selections.
    if active_key is not None and active_key not in set(visible.student_key):
        active_key=None
    render_dashboard_chat(role,principal,route,choice,visible,active_key)
