from datetime import date
from awm_audit.common import query
from awm_audit.oracle import Rule,TaskSpec,call,mutate,row_rule,scalar,sqlref,one,resultref


def build(i,db):
    program=lambda name,inst:scalar(db,'SELECT p.id FROM programs p JOIN institutions x ON x.id=p.institution_id WHERE p.name=? AND x.name=?',(name,inst))
    cycle=lambda name:scalar(db,'SELECT id FROM admission_cycles WHERE name=?',(name,))
    pc=lambda name,inst,cy:scalar(db,'SELECT id FROM program_cycles WHERE program_id=? AND cycle_id=?',(program(name,inst),cycle(cy)))
    latest=lambda t:sqlref(f'SELECT max(id) FROM {t}')
    if i==0:
        p=program('MSc Data Science','Northbridge University');cy=cycle('Fall 2026')
        calls=[call('create_program_cycle',payload={'program_id':p,'cycle_id':cy,'application_deadline':'2026-03-31','capacity':60})]+[call('create_checklist_definition',program_cycle_id=latest('program_cycles'),payload={'name':n,'is_required':1,'sort_order':j+1}) for j,n in enumerate(['Official Transcript','GRE Score Report','One Academic Reference'])]
        rules=[row_rule('program_cycles',{'program_id':p,'cycle_id':cy,'application_deadline':'2026-03-31','capacity':60},new=True),Rule('exact requested required checklist',"SELECT count(*)=3 AND sum(name IN ('Official Transcript','GRE Score Report','One Academic Reference') AND is_required=1)=3 FROM checklist_item_definitions WHERE program_cycle_id=(SELECT max(id) FROM program_cycles)")]
        return TaskSpec(calls,rules,[mutate(0,'payload.capacity',50),{'kind':'omit','index':3,'name':'missing_reference_checklist'}],status='confirmed_defect' if query(db,'SELECT id FROM program_cycles WHERE program_id=? AND cycle_id=?',(p,cy)) else 'ready',notes='明确要求新建的program/cycle组合初态已存在，唯一约束阻止重复创建。',evidence={'existing':query(db,'SELECT * FROM program_cycles WHERE program_id=? AND cycle_id=?',(p,cy)),'constraint':'program_cycles(program_id,cycle_id) UNIQUE'})
    if i==1:
        p=program('Bachelor of Mechanical Engineering','Lakeside Institute')
        existing=query(db,"SELECT id FROM eligibility_rules WHERE program_id=? AND field_name='high_school_math_score'",(p,))
        calls=[call('update_program',program_id=p,payload={'duration_years':4})]+[call('delete_eligibility_rule',rule_id=r['id']) for r in existing]
        body={'rule_type':'minimum_score','description':'High school math score at least 80 percent','field_name':'high_school_math_score','operator':'>=','value':'80'}
        calls.append(call('create_program_eligibility_rule',program_id=p,payload=body))
        return TaskSpec(calls,[row_rule('programs',{'id':p,'duration_years':4}),row_rule('eligibility_rules',{'program_id':p,'field_name':'high_school_math_score','operator':'>=','value':'80'})],[mutate(0,'payload.duration_years',3),mutate(len(calls)-1,'payload.value','70')])
    if i==2:
        program_cycle=pc('MBA - Full Time','Metro Business School','Fall 2025');applicant=one(db,'SELECT * FROM applicants WHERE user_id=1')
        edu=dict(degree_level='bachelor',field_of_study='Economics',graduation_year=2022,institution_name='Not specified')
        calls=[call('create_applicant_education',payload=edu),call('create_application',payload={'program_cycle_id':program_cycle,'status':'in-progress'}),call('create_application_demographics',application_id=latest('applications'),payload={'date_of_birth':applicant['date_of_birth'],'nationality':applicant['country_of_citizenship']}),call('create_application_education_link',application_id=latest('applications'),payload={'education_id':latest('applicant_education'),'is_primary':1})]
        rules=[row_rule('applications',{'applicant_id':applicant['id'],'program_cycle_id':program_cycle,'status':'in-progress'},new=True),Rule('economics degree and demographics linked to new draft',"SELECT EXISTS(SELECT 1 FROM application_education_links l JOIN applicant_education e ON e.id=l.education_id WHERE l.application_id=a.id AND lower(e.degree_level) IN ('bachelor',\"bachelor's\") AND e.field_of_study='Economics' AND e.graduation_year=2022) AND EXISTS(SELECT 1 FROM application_demographics d WHERE d.application_id=a.id AND d.date_of_birth=? AND d.nationality=?) FROM applications a WHERE a.id=(SELECT max(id) FROM applications)",[applicant['date_of_birth'],applicant['country_of_citizenship']])]
        return TaskSpec(calls,rules,[mutate(1,'payload.status','submitted'),mutate(0,'payload.graduation_year',2021)],notes='用户明确要求新建申请，但该申请人/招生周期的申请已存在，唯一约束阻止重复创建。',status='confirmed_defect' if query(db,'SELECT id FROM applications WHERE applicant_id=? AND program_cycle_id=?',(applicant['id'],program_cycle)) else 'ready',evidence={'existing':query(db,'SELECT * FROM applications WHERE applicant_id=? AND program_cycle_id=?',(applicant['id'],program_cycle)),'constraint':'applications(applicant_id,program_cycle_id) UNIQUE'})
    if i==3:
        app=scalar(db,'SELECT a.id FROM applications a JOIN applicants u ON u.id=a.applicant_id WHERE a.program_cycle_id=? AND u.user_id=1',(pc('MSc Computer Science','Horizon University','Fall 2025'),))
        calls=[];rules=[]
        for title,when in [('Passport','2025-01-15'),('English Proficiency Test Score','2025-01-20')]:
            dt=scalar(db,'SELECT id FROM document_types WHERE name=?',(title,))
            calls.append(call('create_document',application_id=app,payload={'document_type_id':dt,'issued_date':when,'file_name':title+'.pdf','file_path':'simulated/'+title+'.pdf'}))
            rules.append(row_rule('documents',{'application_id':app,'document_type_id':dt,'issued_date':when},new=True))
        return TaskSpec(calls,rules,[mutate(0,'payload.issued_date','2025-01-16'),{'kind':'omit','index':1,'name':'missing_english_test'}])
    if i==4:
        p=program('Postgraduate Certificate in Project Management','Central City College');pcy=pc('Postgraduate Certificate in Project Management','Central City College','Spring 2025')
        calls=[call('create_review_workflow',program_id=p,payload={'name':'Audit three-stage review'})]+[call('create_review_step',workflow_id=latest('review_workflows'),payload={'name':name,'step_order':j+1,'is_active':1}) for j,name in enumerate(['Eligibility Check','Faculty Review','Interview'])]+[call('attach_workflow_to_program_cycle',program_cycle_id=pcy,payload={'workflow_id':latest('review_workflows'),'is_default':1})]
        rules=[row_rule('review_workflows',{'program_id':p},new=True),Rule('three active ordered review steps',"SELECT count(*)=3 AND sum((name='Eligibility Check' AND step_order=1) OR (name='Faculty Review' AND step_order=2) OR (name='Interview' AND step_order=3))=3 AND min(is_active)=1 FROM review_steps WHERE workflow_id=(SELECT max(id) FROM review_workflows)"),Rule('cycle default workflow',"SELECT EXISTS(SELECT 1 FROM program_cycle_workflows WHERE program_cycle_id=? AND workflow_id=(SELECT max(id) FROM review_workflows) AND is_default=1)",[pcy])]
        return TaskSpec(calls,rules,[mutate(2,'payload.name','Skipped Faculty'),{'kind':'omit','index':4,'name':'unassigned_workflow'}])
    if i==5:
        step=scalar(db,"SELECT s.id FROM review_steps s JOIN review_workflows w ON w.id=s.workflow_id JOIN program_cycles pc ON pc.program_id=w.program_id JOIN applications a ON a.program_cycle_id=pc.id WHERE a.id=48291 AND s.name='Faculty Review'")
        body=dict(application_id=48291,review_step_id=step,score=88,max_score=100,comments='Strong academic background but limited work experience',status='completed')
        return TaskSpec([call('create_application_review',payload=body)],[row_rule('application_reviews',body,new=True)],[mutate(0,'payload.score',78),mutate(0,'payload.status','pending')])
    if i==6:
        pcy=pc('BSc Nursing','Riverside University','Fall 2025')
        ids=[x['id'] for x in query(db,"SELECT a.id FROM applications a JOIN application_reviews r ON r.application_id=a.id WHERE a.program_cycle_id=? GROUP BY a.id HAVING avg(100.0*r.score/r.max_score)>=85 ORDER BY a.id",(pcy,))]
        calls=[];rules=[]
        for a in ids:
            body={'application_id':a,'decision_status':'accepted','scholarship_amount':2000,'response_deadline':'2025-07-15'}
            existing=query(db,'SELECT * FROM application_decisions WHERE application_id=?',(a,))
            if existing:
                calls.append(call('get_application_decision',application_id=a))
            else:
                calls.append(call('create_application_decision',payload=body))
            rules.append(row_rule('application_decisions',body))
        if not ids:rules=[Rule('no qualifying new decisions',"SELECT count(*)=0 FROM application_decisions WHERE id NOT IN (SELECT id FROM initial.application_decisions)")]
        return TaskSpec(calls,rules,[],notes='按score/max_score归一化；合格申请的正确decision初态已全部存在时允许查询确认，无操作不能强标负例；无足够可经工具产生的负例时覆盖分母不计入。',evidence={'eligible_ids':ids,'existing_decisions':query(db,'SELECT * FROM application_decisions WHERE application_id IN (SELECT value FROM json_each(?))',(__import__('json').dumps(ids),))})
    if i==7:
        pcy=pc('BSc Computer Engineering','Greenfield Tech','Fall 2025')
        app=one(db,'SELECT a.id,d.response_deadline FROM applications a JOIN applicants u ON u.id=a.applicant_id JOIN application_decisions d ON d.application_id=a.id WHERE u.user_id=1 AND a.program_cycle_id=? AND d.decision_status=\'accepted\'',(pcy,))
        if app['response_deadline'] and date.today()>date.fromisoformat(app['response_deadline'][:10]):
            return TaskSpec([],[],status='confirmed_defect',notes='用户要求在答复截止日前确认，但当前运行日期已超过初态offer截止日；原始日期不回拨。',evidence={'current_date':date.today().isoformat(),'offer':app})
        return TaskSpec([],[],status='needs_review',notes='需唯一确认对应offer后才能执行确认与入学，禁止猜测offer ID。',evidence=app)
    if i==8:
        return TaskSpec([],[],status='needs_review',notes='用户要求Summit University的2025-2026招生周期，但需确认多个学期周期如何合并，以及零申请状态是否补0；官方API只接受一个cycle_id。',evidence={'institution':'Summit University','academic_year':'2025-2026','statuses':['in-progress','submitted','under review','waitlisted','accepted','rejected']})
    if i==9:
        p=program('MBA Online','Metro Business School');cy=cycle('Spring 2025')
        apps=query(db,'SELECT a.id,a.applicant_id,a.program_cycle_id FROM applications a JOIN program_cycles pc ON pc.id=a.program_cycle_id WHERE pc.program_id=? AND pc.cycle_id=? ORDER BY a.id',(p,cy));expected=[]
        for a in apps:
            missing=query(db,'SELECT d.id checklist_definition_id,d.name FROM checklist_item_definitions d WHERE d.program_cycle_id=? AND d.is_required=1 AND NOT EXISTS(SELECT 1 FROM application_checklist_items x WHERE x.application_id=? AND x.checklist_definition_id=d.id AND x.is_completed=1) ORDER BY d.id',(a['program_cycle_id'],a['id']))
            rejected=query(db,"SELECT d.id document_id,t.name document_type_name,d.verification_comments FROM documents d JOIN document_types t ON t.id=d.document_type_id WHERE d.application_id=? AND d.verification_status='rejected' ORDER BY d.id",(a['id'],))
            if missing or rejected:expected.append({'application_id':a['id'],'applicant_id':a['applicant_id'],'missing_required_checklist_items':missing,'rejected_documents':rejected})
        return TaskSpec([call('report_qa_missing_or_rejected_checklist',program_id=p,cycle_id=cy)],[],[{'kind':'answer','value':'{}','name':'missing_details'},{'kind':'answer','value':'{"applications":[]}','name':'empty_results'}],answer=resultref(),answer_expected=expected,answer_path=['applications'])
    raise ValueError(i)
