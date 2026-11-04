"""These recipes remain unexecuted while the released initial data is invalid."""
import json
from awm_audit.oracle import Rule,TaskSpec,call,mutate,row_rule,scalar,sqlref,one,resultref


def build(i,db):
    wid=scalar(db,"SELECT id FROM workspaces WHERE name='Acme Automation'")
    workflow=lambda n:scalar(db,'SELECT id FROM workflows WHERE workspace_id=? AND name=?',(wid,n))
    conn=lambda n:scalar(db,'SELECT id FROM connections WHERE workspace_id=? AND name=?',(wid,n))
    latest=lambda t:sqlref(f'SELECT max(id) FROM {t}')
    if i==0:
        body=dict(name='Salesforce Prod',type='crm',api_base_url='https://api.salesforce.com')
        creds=dict(client_id='sf_client_123',client_secret='sf_secret_456',refresh_token='sf_refresh_789')
        return TaskSpec([call('create_connection',workspace_id=wid,body=body),call('create_connection_oauth2_credentials',connection_id=latest('connections'),body=creds)],[row_rule('connections',{'workspace_id':wid,**body},new=True),row_rule('connection_credentials_oauth2',creds,where='connection_id=(SELECT max(id) FROM connections)',new=True)],[mutate(0,'body.type','marketing'),{'kind':'omit','index':1,'name':'missing_credentials'}])
    if i in [1,4]:
        name='New Lead Welcome Email' if i==1 else 'High-Value Lead Escalation'
        body=dict(name=name,status='draft' if i==1 else 'active',execution_policy='sequential')
        calls=[call('create_workflow',workspace_id=wid,body=body),call('create_workflow_trigger',workflow_id=latest('workflows'),body={'type':'record_created','connection_id':conn('Salesforce Prod'),'object_name':'Lead','enabled':True})]
        if i==1:
            config={'from':'welcome@mycompany.com','subject':'Thanks for your interest in Acme!','body':'Hi {{trigger.lead.first_name}} at {{trigger.lead.company}}!'}
            calls.append(call('append_workflow_step',workflow_id=latest('workflows'),body={'label':'Welcome','type':'send_email','action_config_json':config,'input_mapping_json':{}}))
        else:
            config={'title':'High-value lead: {{lead.name}}','priority':'high','description':'{{lead.email}} {{lead.annual_revenue}}'}
            calls += [call('append_workflow_step',workflow_id=latest('workflows'),body={'label':'Create Ticket','type':'create_record','connection_id':conn('Zendesk'),'action_config_json':config,'input_mapping_json':{}}),call('append_workflow_step',workflow_id=latest('workflows'),body={'label':'Notify manager','type':'send_email','action_config_json':{'to':'sales_manager@mycompany.com','body':'Ticket {{step.1.id}} for {{lead.name}} {{lead.email}} {{lead.annual_revenue}}'},'input_mapping_json':{}})]
        rules=[row_rule('workflows',{'workspace_id':wid,**body},new=True),row_rule('workflow_triggers',{'type':'record_created','connection_id':conn('Salesforce Prod'),'object_name':'Lead','enabled':1},where='workflow_id=(SELECT max(id) FROM workflows)'),Rule('first step config and position',"SELECT count(*)=1 FROM workflow_steps WHERE workflow_id=(SELECT max(id) FROM workflows) AND position=1 AND type=? AND json(action_config_json)=json(?)",['send_email' if i==1 else 'create_record',json.dumps(config)])]
        if i==4:rules.append(Rule('manager summary second step',"SELECT count(*)=1 FROM workflow_steps WHERE workflow_id=(SELECT max(id) FROM workflows) AND position=2 AND type='send_email' AND json_extract(action_config_json,'$.to')='sales_manager@mycompany.com' AND action_config_json LIKE '%step.1.id%' AND action_config_json LIKE '%lead.email%'"))
        return TaskSpec(calls,rules,[mutate(0,'body.status','archived'),{'kind':'omit','index':2,'name':'missing_steps'}])
    if i==2:
        target=workflow('Sync Open Support Tickets');body=dict(name='Weekday Hourly Sync',schedule_interval='0 * * * 1-5',time_zone='America/New_York')
        return TaskSpec([call('create_schedule',workspace_id=wid,body=body),call('create_workflow_trigger',workflow_id=target,body={'type':'schedule_interval','schedule_id':latest('schedules'),'enabled':True})],[row_rule('schedules',{'workspace_id':wid,**body},new=True),row_rule('workflow_triggers',{'workflow_id':target,'type':'schedule_interval','enabled':1},where='schedule_id=(SELECT max(id) FROM schedules)',new=True)],[mutate(0,'body.schedule_interval','0 * * * *'),{'kind':'omit','index':1,'name':'missing_trigger'}])
    if i==3:
        target=workflow('Customer Onboarding')
        config={'channel':'#onboarding-alerts','payload':{'name':'{{customer.name}}','email':'{{customer.email}}','onboarding_stage':'{{customer.onboarding_stage}}'}}
        body=dict(after_label='Create CRM Contact',label='Onboarding Slack notice',type='post_message',connection_id=conn('Slack Support'),action_config_json=config,input_mapping_json={})
        return TaskSpec([call('insert_workflow_step_after_label',workflow_id=target,body=body)],[row_rule('workflow_steps',{'workflow_id':target,'label':body['label'],'type':'post_message','connection_id':conn('Slack Support'),'action_config_json':config},new=True),Rule('inserted immediately after anchor',"SELECT count(*)=1 FROM workflow_steps s JOIN workflow_steps a ON a.workflow_id=s.workflow_id AND a.label='Create CRM Contact' WHERE s.workflow_id=? AND s.label='Onboarding Slack notice' AND s.position=a.position+1",[target])],[mutate(0,'body.type','send_email'),mutate(0,'body.action_config_json.channel','#general')])
    if i==5:
        target=workflow('Support Ticket Auto-Responder');step=scalar(db,'SELECT id FROM workflow_steps WHERE workflow_id=? AND position=2',(target,))
        conditions={'conditions':[{'path':'trigger.ticket.priority','operator':'equals','value':'low'},{'path':'trigger.ticket.status','operator':'equals','value':'new'}]}
        return TaskSpec([call('update_workflow_step_conditions',step_id=step,body={'logic_type':'all','conditions_json':conditions})],[row_rule('workflow_step_conditions',{'workflow_step_id':step,'logic_type':'all','conditions_json':conditions})],[mutate(0,'body.logic_type','any'),mutate(0,['body','conditions_json','conditions',0,'value'],'high')])
    if i==6:
        return TaskSpec([],[],status='needs_review',notes='需要核对执行器是否真正推进run到终态，以及run_logs完整性；原始run_logs的INSERT语法错误使此题无法进行有效主评测。',evidence={'workflow':'Invoice Payment Reminder','input_payload':{'invoice_id':'INV-2025-00123','customer_email':'jane.doe@example.com','due_date':'2025-01-15','amount':499}})
    if i==7:
        target=workflow('Bulk Lead Import Processor')
        return TaskSpec([call('update_workspace_rate_limit',workspace_id=wid,body={'rate_limit_max_runs_per_minute':200}),call('update_workflow_concurrency_control',workflow_id=target,body={'max_concurrent_runs':10})],[row_rule('workspaces',{'id':wid,'rate_limit_max_runs_per_minute':200}),row_rule('workflow_concurrency_controls',{'workflow_id':target,'max_concurrent_runs':10})],[mutate(0,'body.rate_limit_max_runs_per_minute',500),mutate(1,'body.max_concurrent_runs',20)])
    if i==8:
        return TaskSpec([],[],status='needs_review',notes='克隆须独立核对全部触发器、步骤、条件及额外收件人，原始环境初始化错误，不能把仅有新workflow行当作完整克隆已验证。',evidence={'source':'Weekly Pipeline Digest','clone':'Weekly Pipeline Digest v2','required_status':'draft','required_execution_policy':'parallel','additional_recipient':'director.sales@mycompany.com'})
    if i==9:
        return TaskSpec([],[],status='needs_review',notes='重放须核对最新失败run原始payload和仅覆盖目标step list_id，原始run_logs缺失使调试步骤证据不完整。',evidence={'workflow':'Sync Contacts to Marketing Tool','override_list_id':'prospects-q1-2025'})
    raise ValueError(i)
