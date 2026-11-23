"""Access-scoped academic dashboard Q&A with optional generative responses.

Rule-based answers work offline. The opt-in AI adapter sends only the already
visible, de-identified dashboard summary, never the full student database.
"""
from __future__ import annotations

import json
import re
from urllib.request import Request, urlopen

import pandas as pd

from src.portal_data import credit_status


def _number(value: object, digits: int = 2) -> float | None:
    numeric = pd.to_numeric(value, errors='coerce')
    return None if pd.isna(numeric) else round(float(numeric), digits)


def build_dashboard_context(view: pd.DataFrame, ml: pd.DataFrame,
                            courses: pd.DataFrame, curriculum: pd.DataFrame,
                            options: pd.DataFrame | None = None, *,
                            student_key: str | None = None,
                            dashboard: str = 'Overall Cohort',
                            programme_filter: str = 'All courses') -> dict:
    """Restrict every answer to the visible role-authorised students.

    Excludes matriculation numbers, access codes, addresses, names and emails;
    those values are unnecessary for dashboard/academic Q&A.
    """
    allowed = set(view.student_key.astype(str)) if 'student_key' in view else set()
    scope = view[view.student_key.astype(str).isin(allowed)] if allowed else view.iloc[0:0]
    counts = scope.programme_code.value_counts().to_dict() if 'programme_code' in scope else {}
    risk = pd.to_numeric(scope.get('risk_probability', pd.Series(dtype=float)), errors='coerce')
    programme_summary = {}
    if 'programme_code' in scope:
        for programme, group in scope.groupby('programme_code'):
            covered = pd.to_numeric(group.get('risk_probability', pd.Series(index=group.index, dtype=float)), errors='coerce')
            programme_summary[str(programme)] = {
                'students': int(len(group)),
                'average_gpa': _number(group.gpa.mean()) if 'gpa' in group else None,
                'average_cgpa': _number(group.cgpa.mean()) if 'cgpa' in group else None,
                'at_risk': int(group.predicted_at_risk.eq(True).sum()) if 'predicted_at_risk' in group else 0,
                'prediction_count': int(covered.notna().sum()),
                'failed_attempts': int(group.failed_courses_all.sum()) if 'failed_courses_all' in group else 0,
            }
    standing_counts = (scope.standing.fillna('Not recorded').replace('', 'Not recorded')
                       .value_counts().to_dict()) if 'standing' in scope else {}
    context = {
        'dashboard': dashboard,
        'programme_filter': programme_filter,
        'student_count': int(scope.student_key.nunique()),
        'programme_counts': {str(k): int(v) for k, v in counts.items()},
        'programme_summary': programme_summary,
        'academic_standing_counts': {str(k):int(v) for k,v in standing_counts.items()},
        'average_gpa': _number(scope.gpa.mean()) if 'gpa' in scope and not scope.empty else None,
        'average_cgpa': _number(scope.cgpa.mean()) if 'cgpa' in scope and not scope.empty else None,
        'students_at_risk': int(scope.predicted_at_risk.eq(True).sum()) if 'predicted_at_risk' in scope else 0,
        'risk_prediction_count': int(risk.notna().sum()),
        'risk_threshold_percent': 24,
        'failed_course_attempts': int(scope.failed_courses_all.sum()) if 'failed_courses_all' in scope else None,
    }
    if not student_key or student_key not in allowed:
        return context
    person = scope[scope.student_key.eq(student_key)].iloc[0]
    history = ml[ml.student_key.eq(student_key)].sort_values('semester_seq')
    attempts = courses[courses.student_key.eq(student_key)].sort_values(['semester_seq','module_code'])
    programme = str(person.programme_code)
    credits = credit_status(attempts, curriculum, programme)
    semester_records=[]
    for r in history.itertuples():
        semester_records.append({
            'number': int(r.semester_seq), 'year': int(r.year) if pd.notna(r.year) else None,
            'period': str(r.period), 'gpa': _number(r.gpa), 'cgpa': _number(r.cgpa),
            'standing': str(r.standing) if pd.notna(r.standing) else None,
            'recorded_credit_hours': _number(r.semester_credit_hours, 0),
        })
    course_records=[]
    for r in attempts.itertuples():
        course_records.append({
            'semester': int(r.semester_seq), 'code': str(r.module_code),
            'course': str(r.module), 'grade': str(r.grade) if pd.notna(r.grade) else 'Not recorded',
        })
    context['student'] = {
        'programme': programme,
        'current_semester': int(person.current_semester) if pd.notna(person.current_semester) else None,
        'latest_gpa': _number(person.gpa), 'latest_cgpa': _number(person.cgpa),
        'academic_standing': str(person.standing) if pd.notna(person.standing) else None,
        'risk_probability_percent': _number(person.risk_probability*100, 1),
        'credit_hours': {k:(v if isinstance(v,bool) else _number(v, 0) if isinstance(v,(int,float)) else v)
                         for k,v in credits.items()},
        'semester_history': semester_records, 'course_attempts': course_records,
    }
    return context


def _fmt(value: object, suffix: str = '') -> str:
    return 'not available' if value is None else f'{value}{suffix}'



# Suggested questions are separately curated for offline and optional generative modes.
# Every choice has a corresponding data-grounded offline answer, including if an API fails.
SUGGESTED_PROMPTS = {
    ('individual', 'offline'): [
        ('i_latest', 'What are my latest GPA, CGPA and academic standing?'),
        ('i_change', 'How did my CGPA change from the previous semester?'),
        ('i_best', 'Which semester had my highest GPA?'),
        ('i_worst', 'Which semester had my lowest GPA?'),
        ('i_semester', 'What happened in semester 3?'),
        ('i_failed', 'Which courses did I fail or receive QF in?'),
        ('i_recovered', 'Have I passed any courses I previously failed?'),
        ('i_credits', 'How many credit hours must I still pass?'),
        ('i_risk', 'What is my predicted academic risk?'),
        ('i_pathway', 'What does my future semester pathway contain?'),
        ('i_attempts', 'How many course attempts are on my record?'),
        ('i_busy', 'Which semester had the most course attempts?'),
    ],
    ('individual', 'generative'): [
        ('i_latest', 'Explain my latest academic results in plain language.'),
        ('i_change', 'Summarise the change in my recorded CGPA over time.'),
        ('i_best', 'Highlight my strongest semester using the GPA evidence.'),
        ('i_worst', 'Explain which recorded semester needs the most GPA improvement.'),
        ('i_failed', 'Summarise my failed courses for an academic advisor discussion.'),
        ('i_recovered', 'Identify which earlier failed courses were passed on a later attempt.'),
        ('i_credits', 'Create a fact-based credit completion checklist for me.'),
        ('i_risk', 'Explain what my academic risk prediction does and does not mean.'),
        ('i_pathway', 'Summarise my provisional future pathway and its limitations.'),
        ('i_busy', 'Identify my semester with the heaviest recorded course load.'),
    ],
    ('cohort', 'offline'): [
        ('c_count', 'How many students are enrolled in each programme?'),
        ('c_best', 'Which programme has the highest average CGPA?'),
        ('c_lowest', 'Which programme has the lowest average GPA?'),
        ('c_means', 'What are the overall average GPA and CGPA?'),
        ('c_risk', 'How many students are predicted at risk?'),
        ('c_risk_by_prog', 'What is the risk prediction rate by programme?'),
        ('c_standing', 'Which academic standing occurs most often?'),
        ('c_fail', 'How many failed course attempts are recorded?'),
        ('c_fail_by_prog', 'Which programme has the most failed course attempts?'),
        ('c_missing', 'How many students lack a risk prediction?'),
        ('c_compare', 'How does IT performance compare with CS and IS?'),
        ('c_scope', 'What dashboard and programme filter am I viewing?'),
    ],
    ('cohort', 'generative'): [
        ('c_count', 'Describe the distribution of students across the visible programmes.'),
        ('c_best', 'Explain which programme leads in average CGPA using the dashboard data.'),
        ('c_lowest', 'Highlight the programme with the lowest recorded average GPA.'),
        ('c_risk', 'Prepare a factual academic risk briefing for the visible cohort.'),
        ('c_risk_by_prog', 'Compare the risk prediction rates among programmes.'),
        ('c_standing', 'Summarise the most common academic standings in this cohort.'),
        ('c_fail_by_prog', 'Compare failed-course attempt totals across programmes.'),
        ('c_missing', 'Explain the scope and limitations of available ML risk predictions.'),
        ('c_compare', 'Write a short evidence-based comparison of IT, CS and IS.'),
        ('c_means', 'Explain what the visible cohort average GPA and CGPA indicate.'),
    ],
}


def suggested_prompts(*, individual: bool, generative: bool) -> list[str]:
    """Return questions suited to the active scope and chat mode."""
    scope = 'individual' if individual else 'cohort'
    mode = 'generative' if generative else 'offline'
    return [question for _, question in SUGGESTED_PROMPTS[(scope, mode)]]


_PROMPT_INTENTS = {
    question.strip().casefold(): intent
    for examples in SUGGESTED_PROMPTS.values()
    for intent, question in examples
}


def _suggested_answer(intent: str, context: dict) -> str:
    """Purpose-specific answers to every curated question, restricted to scoped facts."""
    student = context.get('student')
    programme_data = context.get('programme_summary', {})
    if intent.startswith('c_'):
        if intent == 'c_count':
            details = ', '.join(f'{p}: {v["students"]}' for p, v in sorted(programme_data.items())) or 'none'
            return f"There are {context['student_count']} students in the visible cohort. By programme: {details}."
        if intent in ('c_best', 'c_lowest'):
            field = 'average_cgpa' if intent == 'c_best' else 'average_gpa'
            usable = [(name, val[field]) for name, val in programme_data.items() if val[field] is not None]
            if not usable: return 'No recorded programme-level grade-point averages are available.'
            name, score = (max(usable, key=lambda item: item[1]) if intent == 'c_best'
                           else min(usable, key=lambda item: item[1]))
            comparison = ', '.join(f'{p}: {_fmt(v[field])}' for p, v in sorted(programme_data.items()))
            direction = 'highest average CGPA' if intent == 'c_best' else 'lowest average GPA'
            return f'{name} has the {direction} at {score:.2f}. All visible programme averages: {comparison}.'
        if intent == 'c_means':
            return (f"Across {context['student_count']} visible students, the average GPA is "
                    f"{_fmt(context['average_gpa'])} and the average CGPA is {_fmt(context['average_cgpa'])}. "
                    'These are averages of the latest available individual results, not forecasts.')
        if intent == 'c_risk':
            at_risk = context['students_at_risk']
            predicted = context['risk_prediction_count']
            rate = f'{at_risk / predicted:.1%}' if predicted else 'not available'
            return (f'{at_risk} of {predicted} students with ML predictions are classified at risk '
                    f'({rate} of those predicted), based on the approximately 24% decision threshold. '
                    'Students without predictions are excluded from this rate.')
        if intent == 'c_risk_by_prog':
            parts=[]
            for p, v in sorted(programme_data.items()):
                rate = f'{v["at_risk"] / v["prediction_count"]:.1%}' if v['prediction_count'] else 'unavailable'
                parts.append(f'{p}: {v["at_risk"]}/{v["prediction_count"]} predicted at risk ({rate})')
            return 'Programme-specific ML prediction results: ' + ('; '.join(parts) if parts else 'no programme data.')
        if intent == 'c_standing':
            standings = context.get('academic_standing_counts', {})
            if not standings:return 'No academic standing data are available in this view.'
            best = max(standings, key=standings.get)
            details = ', '.join(f'{name}: {count}' for name, count in sorted(standings.items(), key=lambda x:-x[1]))
            return f'The most common recorded academic standing is {best} ({standings[best]} students). Breakdown: {details}.'
        if intent == 'c_fail':
            return (f"There are {_fmt(context.get('failed_course_attempts'))} failed course attempts "
                    'in the visible student scope. This counts attempts rather than unique failed subjects.')
        if intent == 'c_fail_by_prog':
            if not programme_data:return 'No programme-level failure data are available.'
            most=max(v['failed_attempts'] for v in programme_data.values())
            leaders=sorted(p for p, v in programme_data.items() if v['failed_attempts']==most)
            details=', '.join(f'{p}: {v["failed_attempts"]}' for p,v in sorted(programme_data.items()))
            if len(leaders)>1:
                return (f"{' and '.join(leaders)} are tied for the most recorded failed attempts "
                        f'({most} each). Failed attempts by programme: {details}.')
            return (f'{leaders[0]} has the most recorded failed attempts ({most}). '
                    f'Failed attempts by programme: {details}.')
        if intent == 'c_missing':
            missing=context['student_count'] - context['risk_prediction_count']
            return (f'{missing} of {context["student_count"]} visible students lack an ML risk prediction; '
                    f'{context["risk_prediction_count"]} have one. Do not assume that missing predictions imply low risk.')
        if intent == 'c_compare':
            pieces=[f'{p}: {v["students"]} students, average GPA {_fmt(v["average_gpa"])}, '
                    f'average CGPA {_fmt(v["average_cgpa"])}' for p,v in sorted(programme_data.items())]
            return 'Comparison of visible programmes: ' + ('; '.join(pieces) if pieces else 'no programme results.')
        if intent == 'c_scope':
            programmes=', '.join(sorted(programme_data)) or 'none'
            return (f"You are viewing the {context['dashboard']} dashboard with the "
                    f"{context['programme_filter']} programme filter. It covers "
                    f"{context['student_count']} accessible students, across {programmes}.")

    if not student:
        return 'Individual academic details are unavailable in this dashboard scope.'
    history = student.get('semester_history', [])
    attempts = student.get('course_attempts', [])
    numeric = [r for r in history if r.get('gpa') is not None]
    if intent == 'i_latest':
        if not history:return 'No recorded academic results are available.'
        latest=history[-1]
        return (f"Latest recorded Semester {latest['number']}: GPA {_fmt(latest['gpa'])}, "
                f"CGPA {_fmt(latest['cgpa'])}, standing {_fmt(latest['standing'])}. "
                f"The programme is {student['programme']}.")
    if intent == 'i_change':
        cgpa=[r for r in history if r.get('cgpa') is not None]
        if len(cgpa)<2:return 'At least two recorded CGPA results are needed to calculate a change.'
        prior, latest=cgpa[-2:]
        delta=latest['cgpa']-prior['cgpa']
        return (f"CGPA moved from {prior['cgpa']:.2f} in Semester {prior['number']} "
                f"to {latest['cgpa']:.2f} in Semester {latest['number']}, "
                f"a change of {delta:+.2f} grade points.")
    if intent in ('i_best', 'i_worst'):
        if not numeric:return 'No recorded semester GPA is available for comparison.'
        term=(max(numeric,key=lambda r:r['gpa']) if intent=='i_best'
              else min(numeric,key=lambda r:r['gpa']))
        descriptor='highest' if intent=='i_best' else 'lowest'
        return (f"Your {descriptor} recorded GPA is {term['gpa']:.2f}, "
                f"from Semester {term['number']} ({term['period']}, {term['year']}). "
                'This comparison uses recorded, not forecast, semester results.')
    if intent == 'i_semester':
        term=next((r for r in history if r['number']==3),None)
        subjects=[r for r in attempts if r['semester']==3]
        if term is None and not subjects:return 'No recorded details were found for Semester 3.'
        overview=(f"Semester 3: GPA {_fmt(term['gpa'])}, CGPA {_fmt(term['cgpa'])}, "
                  f"standing {_fmt(term['standing'])}. " if term else 'Semester 3 has course records but no GPA result. ')
        return overview + (f'{len(subjects)} course attempts are recorded: ' +
                           '; '.join(f"{r['code']} ({r['grade']})" for r in subjects) if subjects
                           else 'No course attempts were recorded for that term.')
    if intent == 'i_failed':
        failed=[r for r in attempts if r['grade'].upper() in ('F','QF')]
        if not failed:return 'No F or QF course attempts are recorded.'
        return (f'{len(failed)} failed attempts: ' + '; '.join(
            f"{r['course']} ({r['code']}), Semester {r['semester']}, grade {r['grade']}" for r in failed)
            + '. Later passes should be checked separately.')
    if intent == 'i_recovered':
        failed=[r for r in attempts if r['grade'].upper() in ('F','QF')]
        recovered=[]
        for entry in failed:
            later=[r for r in attempts if r['code']==entry['code'] and r['semester']>entry['semester']
                   and r['grade'].upper() not in ('F','QF','NOT RECORDED','NAN','')]
            if later and entry['code'] not in recovered: recovered.append(entry['code'])
        if not failed:return 'No recorded F or QF attempts are available to review for retakes.'
        if not recovered:return (f'{len(failed)} failed attempts are recorded, but no later non-failing '
                                 'grade for the same course code was found in this history.')
        return ('Courses with a recorded F/QF followed by a later non-failing grade: '
                + ', '.join(recovered) + '. Confirm official completion with your advisor.')
    if intent == 'i_credits':
        c=student['credit_hours']
        return (f"{_fmt(c['passed'])} passing credits are recorded against {c['required']} required "
                f"for {student['programme']}; {_fmt(c['credits_to_pass'])} further credits need "
                f"passing grades. {_fmt(c['failed'])} failed and {_fmt(c['remaining'])} "
                'unattempted/remaining credits are shown separately.')
    if intent == 'i_risk':
        return (f"Your predicted risk is {_fmt(student['risk_probability_percent'], '%')} "
                'against an approximately 24% dashboard threshold. This is a model-based early-warning '
                'signal for academic review, not a final academic decision.')
    if intent == 'i_pathway':
        path=context.get('future_pathway')
        if not path:return ('No future pathway is available yet. Use Generate Next Semester Pathway '
                            'to calculate a provisional plan, then ask again.')
        semesters=', '.join(f'Semester {n}' for n in path['semesters']) or 'no subsequent semesters'
        completeness='still needs advisor review' if path['incomplete'] else 'reaches the modelled passing-credit target'
        return (f"The provisional plan shows {semesters}, with {path['planned_passing_credits']:g} "
                f"of {path['required_credits']} planned passing credits; it {completeness}. "
                'Projected passes are hypothetical, not earned grades.')
    if intent == 'i_attempts':
        distinct=len({r['code'] for r in attempts})
        return (f'{len(attempts)} recorded course attempts cover {distinct} distinct course codes. '
                'An attempt is counted separately when the same course is taken again.')
    if intent == 'i_busy':
        if not attempts:return 'No recorded course attempts are available to compare semester loads.'
        counts={}
        for row in attempts:counts[row['semester']]=counts.get(row['semester'],0)+1
        n=max(sorted(counts),key=lambda semester:counts[semester])
        return (f'Semester {n} has the most recorded course attempts ({counts[n]}). '
                'This counts subjects, not credit-hour workload.')
    return 'That question cannot be answered from the visible academic data.'

def answer_dashboard_question(question: str, context: dict) -> str:
    """Offline interactive answers derived solely from accessible dashboard facts."""
    q = question.strip().lower()
    intent = _PROMPT_INTENTS.get(q)
    if intent:
        return _suggested_answer(intent, context)
    student = context.get('student')
    if not q:
        return 'Ask about the semester timeline, academic performance, credit hours, failures, or study pathway.'
    if re.fullmatch(r'(hi|hello|hey|good morning|good evening)[!. ]*', q):
        return 'Hello! Ask me about this dashboard, grades, semesters, failed courses, credit hours, or the future-semester pathway.'
    if any(s in q for s in ['what can you', 'how can you', 'help', 'example questions']):
        return ('I can explain the data shown on this dashboard. Try “What is the latest CGPA?”, '
                '“Which courses failed?”, “What happened in semester 3?”, '
                '“How many credit hours remain?”, or “How many students are at risk?”.')
    if any(s in q for s in ['generative ai', 'ai mode', 'offline mode']):
        return ('With Generative AI switched off, I answer directly from this dashboard’s academic data. '
                'Switch it on and configure OPENAI_API_KEY to use the optional generative model. '
                'AI-generated answers still require academic-advisor verification.')
    if student:
        sem_match = re.search(r'\b(?:semester|sem|term)\s*(\d+)\b', q)
        if sem_match:
            number = int(sem_match.group(1))
            item = next((r for r in student['semester_history'] if r['number']==number),None)
            items = [r for r in student['course_attempts'] if r['semester']==number]
            if item is None and not items:
                return f'No recorded academic details were found for Semester {number} in the accessible data.'
            opening = (f"Semester {number} ({item['period']}, {item['year']}): "
                       f"GPA {_fmt(item['gpa'])}, CGPA {_fmt(item['cgpa'])}, "
                       f"standing {_fmt(item['standing'])}." if item else
                       f'Semester {number} has course records but no GPA/CGPA record.')
            if not items:
                return opening + ' No individual course attempts are recorded for that semester.'
            listing = '; '.join(f"{r['course']} ({r['code']}): {r['grade']}" for r in items)
            return f'{opening}\n\nRecorded courses: {listing}.'
        if any(s in q for s in ['fail','qf','retake','repeat']):
            failed = [r for r in student['course_attempts'] if r['grade'].upper() in ('F','QF')]
            if not failed:
                return 'No F or QF course attempts are recorded in the accessible student history.'
            items = '; '.join(f"{r['course']} ({r['code']}, Semester {r['semester']}, {r['grade']})" for r in failed)
            return f'{len(failed)} F/QF attempts are recorded: {items}. Repeats should be checked against later passing attempts and advisor requirements.'
        if any(s in q for s in ['credit', 'graduat', 'completion','remaining']):
            c=student['credit_hours']
            return (f"{_fmt(c['passed'])} passing credits out of {c['required']} required for "
                    f"{student['programme']}; {_fmt(c['failed'])} failed credits and "
                    f"{_fmt(c['remaining'])} unattempted/remaining credits are shown. "
                    f"Another {_fmt(c['credits_to_pass'])} credits must receive passing grades. "
                    'Planned or unrecorded courses are not verified passes.')
        if any(s in q for s in ['pathway','future','recommend','planned semesters','study plan']):
            future=context.get('future_pathway')
            if not future:
                return 'Generate Next Semester Pathway first to view provisional future-semester course recommendations.'
            semesters = ', '.join(f'Semester {n}' for n in future['semesters']) or 'none beyond the next term'
            status = 'Some requirements still need advisor review.' if future['incomplete'] else 'The modelled passing-credit target is reached.'
            return (f"The displayed future pathway includes {semesters}; its provisional passing-credit total "
                    f"is {future['planned_passing_credits']:g} / {future['required_credits']}. {status} "
                    'These are hypothetical passes, not earned grades.')
        if any(s in q for s in ['risk', 'at risk','prediction']):
            return (f"This student's predicted at-risk probability is "
                    f"{_fmt(student['risk_probability_percent'],'%')} compared with a dashboard "
                    'reference threshold of approximately 24%. It is an advisory ML prediction, not a final decision.')
        if any(s in q for s in ['gpa','cgpa','performance','trend','improv','declin','standing']):
            records=student['semester_history']
            if not records:
                return 'No recorded GPA or CGPA values are available for this student.'
            last=records[-1]
            previous=records[-2] if len(records)>1 else None
            comparison = ''
            if previous and last['cgpa'] is not None and previous['cgpa'] is not None:
                delta=round(last['cgpa']-previous['cgpa'],2)
                comparison=f' CGPA changed by {delta:+.2f} points since the prior recorded semester.'
            return (f"Latest recorded Semester {last['number']}: GPA {_fmt(last['gpa'])}, "
                    f"CGPA {_fmt(last['cgpa'])}, standing {_fmt(last['standing'])}."+comparison)
        if any(s in q for s in ['semester','term', 'current', 'latest']):
            return (f"The current semester is {_fmt(student['current_semester'])}. "
                    f"There are {len(student['semester_history'])} recorded semester performance entries. "
                    'Ask about a specific semester for courses and GPA/CGPA.')
        if any(s in q for s in ['course', 'subject','grade']):
            entries=student['course_attempts']
            return (f'{len(entries)} course attempts are recorded for this student. '
                    'Ask about a specific semester, failed courses, or credit hours for details.')
    if any(s in q for s in ['risk','at risk','prediction']):
        return (f"{context['students_at_risk']} of {context['risk_prediction_count']} students with "
                'available ML predictions are classified at risk in this visible dashboard scope '
                '(approximately 24% threshold).')
    if any(s in q for s in ['gpa','cgpa','performance','average']):
        return (f"The visible {context['student_count']} students have average GPA "
                f"{_fmt(context['average_gpa'])} and average CGPA {_fmt(context['average_cgpa'])}.")
    if any(s in q for s in ['course','fail','qf']):
        return f"The visible dashboard currently contains {_fmt(context['failed_course_attempts'])} failed course attempts across its students."
    if any(s in q for s in ['student','programme','program','dashboard','overview','cohort','filter','how many']):
        counts=', '.join(f'{p}: {n}' for p,n in sorted(context['programme_counts'].items())) or 'no students'
        return (f"This {context['dashboard']} view contains {context['student_count']} students under "
                f"the {context['programme_filter']} filter. Programme counts: {counts}.")
    return ('I can answer questions about the academic data currently visible to your account, including '
            'GPA/CGPA, semesters, course grades, failed courses, credit hours, and the pathway. '
            'Try a more specific question, or enable Generative AI for more conversational explanations.')


def generate_ai_answer(question: str, context: dict, messages: list[dict],
                       api_key: str, *, model: str = 'gpt-4o-mini') -> str:
    """Optional API-based chat, with only scope-limited de-identified context."""
    payload = {
        'model': model,
        'messages': [
            {'role':'system','content':(
                'You are an academic dashboard assistant. Answer using ONLY the supplied '
                'dashboard facts; do not invent scores, credits, courses, academic policies, '
                'eligibility, or future results. The planning pathway is hypothetical. '
                'Never claim a student will definitely pass or graduate. If a fact is absent, '
                'explain the limitation. Do not identify or discuss people outside the '
                'provided scope. Be concise and clear.')},
            {'role':'system','content':'Scoped academic dashboard facts (JSON): '+json.dumps(context,ensure_ascii=False)},
            *[{'role':m['role'],'content':m['content'][:1500]}
              for m in messages[-6:] if m.get('role') in ('user','assistant')],
            {'role':'user','content':question[:1500]},
        ],
        'max_tokens': 500,
        'temperature': 0.2,
    }
    request=Request('https://api.openai.com/v1/chat/completions',
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Authorization':f'Bearer {api_key}',
                             'Content-Type':'application/json'},
                    method='POST')
    with urlopen(request, timeout=25) as response:
        result=json.loads(response.read())
    answer=result['choices'][0]['message']['content']
    if not answer or not str(answer).strip():
        raise ValueError('Empty generative AI response')
    return str(answer).strip()
