from awm_audit.common import query
from awm_audit.oracle import Rule, TaskSpec, call, mutate, one, row_rule, scalar, sqlref, resultref


def build(i, db):
    if i in (2,3,6,7,9):
        owner=one(db,'SELECT id,user_id FROM apprentices WHERE id=552')
        if owner['user_id'] not in (None,1):
            return TaskSpec([],[],status='confirmed_defect',notes='任务指定apprentice 552属于user 2，而相关官方读写工具将当前用户固定为1；真实调用返回apprentice_id=0的占位记录或全零摘要，且无合法工具切换用户或修改所有权。',evidence={'target_owner':owner,'current_user_id':1,'guard':'apprentice.user_id is not None and apprentice.user_id != 1','source':'official generated API; profiles/education_training_management_1.api.json','affected_task':i})
    if i == 0:
        org=scalar(db,"SELECT id FROM organizations WHERE name='Metro Electric Co.'")
        body=dict(organization_id=org,name='Residential Electrician Level 1',duration_months=24,required_hours_on_job=4000)
        pid=sqlref("SELECT max(id) FROM programs WHERE name='Residential Electrician Level 1'")
        calls=[call('create_program',body=body)]+[call('add_program_required_competency',program_id=pid,body={'competency_id':c,'is_mandatory':1}) for c in [101,102,205,310]]
        rules=[row_rule('programs',body,new=True),Rule('all and only requested competencies',"SELECT count(*)=4 AND count(DISTINCT competency_id)=4 AND sum(competency_id IN (101,102,205,310))=4 FROM program_required_competencies WHERE program_id=(SELECT max(id) FROM programs WHERE name='Residential Electrician Level 1')")]
        return TaskSpec(calls,rules,[mutate(0,'body.required_hours_on_job',3000),{'kind':'omit','index':4,'name':'missing_competency'}])
    if i == 1:
        eid=scalar(db,"SELECT id FROM program_enrollments WHERE apprentice_id=552 AND program_id=32")
        demographics=dict(date_of_birth='1999-04-18',gender='female',prior_experience='1 year as HVAC helper')
        return TaskSpec([call('update_program_enrollment',enrollment_id=eid,body={'program_id':32,'progress_status':'in_progress'}),call('update_apprentice',apprentice_id=552,body=demographics)],
                        [row_rule('program_enrollments',{'id':eid,'apprentice_id':552,'program_id':32,'progress_status':'in_progress'}),row_rule('apprentices',{'id':552,**demographics})],
                        [mutate(0,'body.progress_status','completed'),mutate(1,'body.gender','male')],notes='初态可能已满足；no_action若满足独立规则不会算作负例。')
    if i == 2:
        body=dict(competency_id=205,record_date='2025-01-10',rating=4,comments_short='Completed full install with minimal guidance')
        return TaskSpec([call('create_competency_record',apprentice_id=552,body=body)], [row_rule('competency_records',{'apprentice_id':552,**body},new=True)], [mutate(0,'body.rating',2),mutate(0,'body.competency_id',101)])
    if i == 3:
        body=dict(work_date='2025-01-12',job_type='Residential wiring rough-in',location='123 Oak St, Springfield',hours_worked=7.5,supervisor_signature_flag=1)
        return TaskSpec([call('create_work_log_entry',apprentice_id=552,body=body)],[row_rule('work_log_entries',{'apprentice_id':552,**body},new=True)],[mutate(0,'body.hours_worked',6),mutate(0,'body.supervisor_signature_flag',0)])
    if i == 4:
        body=dict(topic='OSHA Electrical Safety Basics',instructor='Jordan Clark',capacity=20)
        cid=sqlref("SELECT max(id) FROM classes WHERE topic='OSHA Electrical Safety Basics'")
        calls=[call('create_class',body=body),call('create_class_session',class_id=cid,body={'start_datetime':'2025-02-05T09:00:00'}),call('create_class_session',class_id=cid,body={'start_datetime':'2025-02-06T09:00:00'}),call('add_program_required_class',program_id=15,body={'class_id':cid,'is_mandatory':1})]
        rules=[row_rule('classes',body,new=True),Rule('two exact sessions and program association',"SELECT (SELECT count(*) FROM class_sessions WHERE class_id=c.id AND datetime(start_datetime) IN ('2025-02-05 09:00:00','2025-02-06 09:00:00'))=2 AND EXISTS(SELECT 1 FROM program_required_classes p WHERE p.class_id=c.id AND program_id=15) FROM classes c WHERE id=(SELECT max(id) FROM classes WHERE topic='OSHA Electrical Safety Basics')")]
        return TaskSpec(calls,rules,[mutate(0,'body.capacity',25),{'kind':'omit','index':3,'name':'missing_program_link'}])
    if i == 5:
        # An enrollment already exists, and the user asks to set that record.
        eid=scalar(db,"SELECT id FROM class_enrollments WHERE apprentice_id=552 AND class_id=88")
        body=dict(status='enrolled',enrollment_date='2025-01-20')
        return TaskSpec([call('update_class_enrollment',enrollment_id=eid,body=body)],[row_rule('class_enrollments',{'id':eid,'apprentice_id':552,'class_id':88,**body})],[mutate(0,'body.status','completed'),mutate(0,'body.enrollment_date','2025-01-21')])
    if i == 6:
        body=dict(apprentice_id=552,score=86,pass_fail='pass',evaluation_date='2025-02-06',comments='Good safety awareness and tool handling')
        return TaskSpec([call('create_class_evaluation',class_id=88,body=body)],[row_rule('class_evaluations',{'class_id':88,**body},new=True)],[mutate(0,'body.score',76),mutate(0,'body.pass_fail','fail')])
    if i == 7:
        expected=dict(apprentice_id=552,start_date='2025-01-01',end_date='2025-03-31',
                      total_work_hours=scalar(db,"SELECT coalesce(sum(hours_worked),0) FROM work_log_entries WHERE apprentice_id=552 AND work_date BETWEEN '2025-01-01' AND '2025-03-31'"),
                      completed_competencies_count=scalar(db,"SELECT count(DISTINCT competency_id) FROM competency_records WHERE apprentice_id=552 AND rating>=3 AND record_date BETWEEN '2025-01-01' AND '2025-03-31'"),
                      average_evaluation_score=scalar(db,"SELECT avg(score) FROM class_evaluations WHERE apprentice_id=552 AND evaluation_date BETWEEN '2025-01-01' AND '2025-03-31'"))
        return TaskSpec([call('get_apprentice_dashboard_summary',apprentice_id=552,start_date='2025-01-01',end_date='2025-03-31',completed_rating_threshold=3)],[],[{'kind':'answer','value':'{}','name':'omitted_statistics'},{'kind':'answer','value':{**expected,'total_work_hours':-1},'name':'wrong_hours'}],answer=resultref(),answer_expected=expected)
    if i == 8:
        body=dict(level='intermediate',category='pipefitting',short_description='Install and test residential copper and PEX water lines')
        return TaskSpec([call('update_competency',competency_id=340,body=body)],[row_rule('competencies',{'id':340,**body})],[mutate(0,'body.level','basic'),mutate(0,'body.category','electrical')])
    if i == 9:
        hours=scalar(db,"SELECT coalesce(sum(hours_worked),0)>=(SELECT required_hours_on_job FROM programs WHERE id=32) FROM work_log_entries WHERE apprentice_id=552")
        missing=query(db,"SELECT competency_id FROM program_required_competencies r WHERE program_id=32 AND is_mandatory=1 AND NOT EXISTS(SELECT 1 FROM competency_records c WHERE c.apprentice_id=552 AND c.competency_id=r.competency_id AND rating>=3)")
        classes=query(db,"SELECT class_id FROM program_required_classes r WHERE program_id=32 AND is_mandatory=1 AND NOT EXISTS(SELECT 1 FROM class_evaluations e WHERE e.apprentice_id=552 AND e.class_id=r.class_id AND lower(pass_fail)='pass')")
        body=dict(apprentice_id=552,program_id=32,issuing_body='State HVAC Licensing Board',issue_date='2025-03-01',expiry_date='2028-03-01',verification_status='verified')
        return TaskSpec([call('get_program_completion_status',program_id=32,apprentice_id=552),call('create_certification',body=body)],
                        [row_rule('certifications',body,new=True),Rule('initial prerequisites satisfied',"SELECT ?",[int(bool(hours) and not missing and not classes)])],
                        [mutate(1,'body.expiry_date','2027-03-01'),mutate(1,'body.program_id',15)],evidence={'hours_met':bool(hours),'missing_competencies':missing,'unpassed_classes':classes})
    raise ValueError(i)
