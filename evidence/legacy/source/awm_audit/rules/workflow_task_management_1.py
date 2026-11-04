import json
from datetime import date
from awm_audit.common import query
from awm_audit.oracle import Rule,TaskSpec,call,mutate,row_rule,scalar,sqlref,one,resultref


def build(i,db):
    cycle=lambda n:scalar(db,'SELECT id FROM application_cycles WHERE name=?',(n,))
    latest=lambda t:sqlref(f'SELECT max(id) FROM {t}')
    if i==0:
        body=dict(name='Community Health Innovation 2026',total_budget=2000000)
        criteria=dict(organization_type='nonprofit',tax_status_required='501(c)(3)',state_required='California',min_annual_budget=250000,max_annual_budget=5000000)
        calls=[call('create_grant_program_endpoint',body=body),call('create_eligibility_criteria_endpoint',body={'grant_program_id':latest('grant_programs'),**criteria}),call('create_application_cycle_endpoint',body={'grant_program_id':latest('grant_programs'),'name':'Community Health Innovation 2026','cycle_year':2026,'open_date':'2026-03-01','close_date':'2026-06-30'})]
        rules=[row_rule('grant_programs',body,new=True),row_rule('eligibility_criteria',criteria,where='grant_program_id=(SELECT max(id) FROM grant_programs)',new=True),row_rule('application_cycles',{'open_date':'2026-03-01','close_date':'2026-06-30'},where='grant_program_id=(SELECT max(id) FROM grant_programs)',new=True)]
        existing=query(db,'SELECT id,name FROM grant_programs WHERE name=?',(body['name'],))
        return TaskSpec(calls,rules,[mutate(0,'body.total_budget',1000000),mutate(1,'body.state_required','Texas')],status='confirmed_defect' if existing else 'ready',notes='用户明确要求新建，但同名program初态已存在且名称唯一；真实工具调用已触发唯一约束。' if existing else '',evidence={'existing':existing,'constraint':'grant_programs.name UNIQUE'})
    if i==1:
        program=scalar(db,"SELECT id FROM grant_programs WHERE name='Youth STEM Enrichment Grants'")
        body=dict(grant_program_id=program,name='Youth STEM Enrichment Grants 2026',cycle_year=2026,open_date='2026-01-05',close_date='2026-02-15',submission_deadline='2026-02-15T23:59:00')
        criteria=dict(organization_type='public school district',state_required='Texas',min_students_served=500)
        return TaskSpec([call('create_application_cycle_endpoint',body=body),call('create_eligibility_criteria_endpoint',body={'cycle_id':latest('application_cycles'),'grant_program_id':program,**criteria})],[row_rule('application_cycles',{k:v for k,v in body.items() if k!='submission_deadline'},new=True),Rule('local deadline',"SELECT datetime(submission_deadline)='2026-02-15 23:59:00' FROM application_cycles WHERE id=(SELECT max(id) FROM application_cycles)"),row_rule('eligibility_criteria',criteria,where='cycle_id=(SELECT max(id) FROM application_cycles)',new=True)],[mutate(0,'body.open_date','2026-01-06'),mutate(1,'body.min_students_served',100)])
    if i==2:
        return TaskSpec([],[],status='needs_review',notes='原任务写“1,000-word character limit”，word和character单位冲突；需确认1000词还是1000字符，不能用官方字段character_limit反推用户意图。',evidence={'cycle':'Arts in Schools 2025','section':'Project Details','required_field':'Community Need Statement','help_text_requirement':'reference local data sources'})
    if i==3:
        cy=cycle('Urban Sustainability Microgrants 2025');org=scalar(db,"SELECT id FROM organizations WHERE name='Green City Partners'")
        calls=[call('update_organization_endpoint',organization_id=org,body={'duns_number':'123456789','annual_operating_budget':1200000}),call('create_application_endpoint',body={'cycle_id':cy,'organization_id':org,'project_title':'Neighborhood Tree Canopy Expansion'})]
        return TaskSpec(calls,[row_rule('organizations',{'id':org,'duns_number':'123456789','annual_operating_budget':1200000}),row_rule('applications',{'cycle_id':cy,'organization_id':org,'project_title':'Neighborhood Tree Canopy Expansion','status':'draft'},new=True)],[mutate(0,'body.duns_number','987654321'),mutate(1,'body.project_title','Wrong project')])
    if i==4:
        a=scalar(db,"SELECT id FROM applications WHERE applicant_user_id=1 AND cycle_id=? AND status='draft'",(cycle('Digital Inclusion Fund 2025'),))
        calls=[call('update_application_endpoint',application_id=a,body={'total_project_budget':150000})]+[call('create_application_budget_item_endpoint',body={'application_id':a,'item_label':n,'amount':v}) for n,v in [('Laptops',80000),('Hotspots',40000),('Digital Literacy Training',30000)]]+[call('submit_application_endpoint',application_id=a)]
        rules=[row_rule('applications',{'id':a,'total_project_budget':150000,'status':'submitted'})]+[row_rule('application_budget_items',{'application_id':a,'item_label':n,'amount':v},new=True) for n,v in [('Laptops',80000),('Hotspots',40000),('Digital Literacy Training',30000)]]
        return TaskSpec(calls,rules,[mutate(0,'body.total_project_budget',140000),{'kind':'omit','index':4,'name':'not_submitted'}])
    if i==5:
        users=[scalar(db,'SELECT id FROM users WHERE username=?',(n,)) for n in ['rsmith_reviewer','ajones_reviewer','kpatel_reviewer']]
        calls=[call('create_review_assignment_endpoint',body={'application_id':4829,'reviewer_user_id':u,'role':role,'review_deadline':'2025-05-10'}) for u,role in zip(users,['primary reviewer','secondary reviewer','secondary reviewer'])]
        rules=[row_rule('review_assignments',{'application_id':4829,'reviewer_user_id':u,'role':role,'review_deadline':'2025-05-10'},new=True) for u,role in zip(users,['primary reviewer','secondary reviewer','secondary reviewer'])]
        return TaskSpec(calls,rules,[mutate(0,'body.role','secondary reviewer'),{'kind':'omit','index':2,'name':'missing_third_reviewer'}])
    if i==6:
        assignment=scalar(db,'SELECT id FROM review_assignments WHERE application_id=5172 AND reviewer_user_id=1')
        body=dict(assignment_id=assignment,application_id=5172,impact_score=4.5,feasibility_score=3.8,budget_realism_score=4,overall_comment='Strong concept but implementation risks need closer monitoring',recommendation='fund with conditions')
        return TaskSpec([call('create_review_endpoint',body=body)],[row_rule('reviews',{**body,'reviewer_user_id':1},new=True)],[mutate(0,'body.impact_score',3.5),mutate(0,'body.recommendation','do not fund')])
    if i==7:
        cy=cycle('Community Libraries 2025');template=scalar(db,"SELECT id FROM notification_templates WHERE name='Shortlist Announcement'")
        ids=[x['id'] for x in query(db,"SELECT id FROM applications WHERE cycle_id=? AND status='under review' AND average_score>4.2 ORDER BY id",(cy,))]
        calls=[];rules=[]
        for a in ids:
            contact=scalar(db,'SELECT u.user_id FROM user_organizations u JOIN applications a ON a.organization_id=u.organization_id WHERE a.id=? AND u.is_primary_contact=1',(a,))
            calls += [call('update_application_endpoint',application_id=a,body={'status':'shortlisted'}),call('create_notification_endpoint',body={'application_id':a,'user_id':contact,'template_id':template,'status':'sent'})]
            rules += [row_rule('applications',{'id':a,'status':'shortlisted'}),row_rule('notifications',{'application_id':a,'user_id':contact,'template_id':template,'status':'sent'},new=True)]
        if not calls:return TaskSpec([],[],status='needs_review',notes='初态没有under review且average_score>4.2的目标申请，不能将无操作当作错误，也不能凭空生成合格申请来验证原任务。',evidence={'cycle_id':cy,'eligible_ids':ids})
        rules.append(Rule('no unrelated application status changes',"SELECT NOT EXISTS(SELECT 1 FROM applications a JOIN initial.applications i ON a.id=i.id WHERE a.status IS NOT i.status AND NOT (i.cycle_id=? AND i.status='under review' AND i.average_score>4.2))",[cy]))
        return TaskSpec(calls,rules,[mutate(0,'body.status','rejected'),{'kind':'omit','index':len(calls)-1,'name':'missing_last_notification'}])
    if i==8:
        cy=cycle('Early Childhood Education 2025')
        apps=query(db,"SELECT id,organization_id,requested_amount FROM applications WHERE cycle_id=? AND status='shortlisted' ORDER BY average_score DESC,id LIMIT 10",(cy,))
        if len(apps)!=10:return TaskSpec([],[],status='confirmed_defect',notes='初态不足10个shortlisted申请，无法满足为前10个创建award且不改动用户未要求的申请状态。',evidence={'shortlisted_count':len(apps),'cycle_id':cy})
        calls=[];rules=[]
        for a in apps:
            amount=min(a['requested_amount'],250000)
            existing=query(db,'SELECT * FROM awards WHERE application_id=?',(a['id'],))
            body=dict(application_id=a['id'],organization_id=a['organization_id'],approved_amount=amount,capped_amount=amount,first_disbursement_date='2025-10-01')
            if existing:
                award=existing[0]['id']
            else:
                calls.append(call('create_award_endpoint',body={**body,'disbursement_schedule':'quarterly over 2 years','status':'approved'}))
                award=latest('awards')
            rules.append(row_rule('awards',body))
            for j in range(8):
                month=10+3*j;when=f'{2025+(month-1)//12}-{(month-1)%12+1:02d}-01'
                share=round(amount/8,2) if j<7 else round(amount-7*round(amount/8,2),2)
                found=query(db,'SELECT * FROM award_disbursements WHERE award_id=? AND date(scheduled_date)=?',(award,when)) if isinstance(award,int) else []
                if not found:
                    calls.append(call('create_award_disbursement_endpoint',body={'award_id':award,'scheduled_date':when,'amount':share,'status':'scheduled'}))
                rules.append(Rule('each quarterly date has exactly one installment and correct amount',"SELECT count(*)=1 AND abs(sum(amount)-?)<0.001 FROM award_disbursements WHERE award_id=(SELECT id FROM awards WHERE application_id=?) AND date(scheduled_date)=?",[share,a['id'],when]))
            rules.append(Rule('exact eight installment schedule',"SELECT count(*)=8 AND abs(sum(amount)-?)<0.001 FROM award_disbursements WHERE award_id=(SELECT id FROM awards WHERE application_id=?)",[amount,a['id']]))
        return TaskSpec(calls,rules,[mutate(0,'body.amount',100),{'kind':'omit','index':len(calls)-1,'name':'missing_final_installment'}],notes='已存在的合规award保留；只补齐缺失季度，不把重复新建错误当作任务无解。')
    if i==9:
        cy=cycle('Housing Stability Innovation Fund 2024');groups={}
        for a in query(db,"SELECT * FROM applications WHERE cycle_id=? AND final_status='awarded' ORDER BY id",(cy,)):
            score=one(db,'SELECT avg(impact_score) impact_score_avg,avg(feasibility_score) feasibility_score_avg,avg(budget_realism_score) budget_realism_score_avg,count(*) review_count FROM reviews WHERE application_id=?',(a['id'],))
            item={k:a[k] for k in ['cycle_id','final_status','project_title','average_score']};item['application_id']=a['id'];item['score_summary']=score
            groups.setdefault(a['organization_id'],[]).append(item)
        expected=[{'organization_id':org,'applications':apps} for org,apps in sorted(groups.items())]
        return TaskSpec([call('get_cycle_submissions_with_scores_by_org_endpoint',cycle_id=cy)],[],[{'kind':'answer','value':'{}','name':'missing_submissions'},{'kind':'answer','value':'{"organizations":[]}','name':'empty_awards'}],answer=resultref(),answer_expected=expected,answer_path=['organizations'])
    raise ValueError(i)
