from datetime import date
from awm_audit.common import query
from awm_audit.oracle import Rule,TaskSpec,call,mutate,row_rule,scalar,sqlref,one


def build(i,db):
    loc=lambda n:scalar(db,'SELECT id FROM locations WHERE name=?',(n,))
    last=lambda t:sqlref(f'SELECT max(id) FROM {t}')
    if i==0:
        location=loc('New York HQ – 5th Floor');user=scalar(db,"SELECT id FROM users WHERE email='jane.smith@acme.com'")
        body=dict(serial_number='LTP-44521',asset_type='Dell Latitude 7440',location_id=location,warranty_expiry='2026-03-31',status='in_use')
        exists=query(db,"SELECT id,serial_number FROM assets WHERE serial_number='LTP-44521'")
        return TaskSpec([call('create_asset',body=body),call('create_asset_assignment',body={'asset_id':last('assets'),'user_id':user,'is_primary':True})],[row_rule('assets',body,new=True),Rule('new laptop primary assignment',"SELECT count(*)=1 FROM asset_assignments a JOIN assets s ON s.id=a.asset_id WHERE s.id NOT IN (SELECT id FROM initial.assets) AND s.serial_number='LTP-44521' AND a.user_id=? AND a.is_primary=1 AND a.unassigned_at IS NULL",[user])],[mutate(0,'body.status','in_stock'),{'kind':'omit','index':1,'name':'missing_assignment'}],status='confirmed_defect' if exists else 'ready',notes='新建序列号在初态已存在且serial_number有唯一约束。' if exists else '',evidence={'existing':exists,'constraint':'assets.serial_number UNIQUE'})
    if i==1:
        body=dict(category='network',impact_level='high',short_description='All users on the 3rd floor at London Office cannot connect to the corporate Wi-Fi SSID FIELD-FIX since 09:10 AM GMT')
        return TaskSpec([call('create_incident',body=body)],[row_rule('incidents',{**body,'lifecycle_status':'logged'},new=True)],[mutate(0,'body.impact_level','low'),mutate(0,'body.category','hardware')])
    if i==2:
        incident=one(db,"SELECT * FROM incidents WHERE category='hardware' AND impact_level='medium' AND lifecycle_status NOT IN ('resolved','closed','cancelled') AND location_id=? ORDER BY reported_at,id LIMIT 1",(loc('Chicago Warehouse'),))
        tech=scalar(db,"SELECT t.id FROM technicians t JOIN users u ON u.id=t.user_id WHERE u.full_name='Carlos Ruiz'")
        body=dict(source_incident_id=incident['id'],asset_id=incident['asset_id'],location_id=incident['location_id'],status='scheduled',assigned_technician_id=tech,scheduled_start='2025-01-02T09:00:00',scheduled_end='2025-01-02T11:00:00')
        calls=[call('create_work_order',body=body),call('create_work_order_part',work_order_id=last('work_orders'),body={'description':'24-port PoE switch','quantity':1}),call('create_work_order_part',work_order_id=last('work_orders'),body={'description':'Cat6 patch cables','quantity':2})]
        rules=[row_rule('work_orders',{k:v for k,v in body.items() if not k.startswith('scheduled_')},new=True),Rule('time slot and required parts',"SELECT datetime(scheduled_start)='2025-01-02 09:00:00' AND datetime(scheduled_end)='2025-01-02 11:00:00' AND EXISTS(SELECT 1 FROM work_order_parts WHERE work_order_id=w.id AND description='24-port PoE switch' AND quantity=1) AND EXISTS(SELECT 1 FROM work_order_parts WHERE work_order_id=w.id AND description='Cat6 patch cables' AND quantity=2) FROM work_orders w WHERE id=(SELECT max(id) FROM work_orders)")]
        return TaskSpec(calls,rules,[mutate(1,'body.quantity',2),{'kind':'omit','index':2,'name':'missing_cables'}])
    if i==3:
        wo=one(db,"SELECT id,assigned_technician_id FROM work_orders WHERE work_order_number='WO-2379'")
        calls=[call('create_work_order_update',work_order_id=wo['id'],body={'update_type':'diagnostic','update_text':'Initial diagnostics performed.'}),call('create_work_order_update',work_order_id=wo['id'],body={'update_type':'repair','update_text':'Replaced defective part.'})]+[call('create_time_entry',work_order_id=wo['id'],body={'technician_id':wo['assigned_technician_id'],'entry_type':kind,'hours':hours,'entry_date':date.today().isoformat()}) for kind,hours in [('onsite_travel',1),('repair_work',1.5)]]+[call('create_work_order_proof',work_order_id=wo['id'],body={'photos_count':3,'customer_acknowledgment_present':True,'metadata_json':'{"signed_customer_acknowledgment":true}'})]
        rules=[Rule('two new diagnostics and repair updates',"SELECT count(*)=2 AND sum(update_type='diagnostic')=1 AND sum(update_type='repair')=1 FROM work_order_updates t WHERE work_order_id=? AND id NOT IN (SELECT id FROM initial.work_order_updates)",[wo['id']]),Rule('new travel and repair time exactly 2.5 hours',"SELECT count(*)=2 AND sum(hours)=2.5 AND sum(entry_type='onsite_travel' AND hours=1)=1 AND sum(entry_type='repair_work' AND hours=1.5)=1 FROM time_entries WHERE work_order_id=? AND id NOT IN (SELECT id FROM initial.time_entries)",[wo['id']]),row_rule('work_order_proof',{'work_order_id':wo['id'],'photos_count':3,'customer_acknowledgment_present':1},new=True)]
        # Upload is an upsert; the user does not require a new proof row.
        rules[-1]=row_rule('work_order_proof',{'work_order_id':wo['id'],'photos_count':3,'customer_acknowledgment_present':1})
        return TaskSpec(calls,rules,[mutate(2,'body.hours',2),mutate(4,'body.photos_count',2)])
    if i==4:
        incident=scalar(db,"SELECT id FROM incidents WHERE incident_number='INC-8842'")
        note='Waiting for vendor RMA approval for replacement firewall hardware'
        return TaskSpec([call('update_incident_status',incident_id=incident,body={'new_status':'on_hold'}),call('create_incident_note',incident_id=incident,body={'note_type':'internal','note_text':note,'expected_update_date':'2025-01-07'})],[row_rule('incidents',{'id':incident,'lifecycle_status':'on_hold'}),row_rule('incident_notes',{'incident_id':incident,'note_type':'internal','note_text':note,'expected_update_date':'2025-01-07'},new=True)],[mutate(0,'body.new_status','closed'),mutate(1,'body.expected_update_date','2025-01-08')])
    if i==5:
        body=dict(title='Upgrade core router firmware at Dallas Datacenter',location_id=loc('Dallas Datacenter'),risk_level='high',implementation_window_start='2025-02-14T23:00:00',implementation_window_end='2025-02-15T02:00:00',rollback_plan='revert to firmware v12.1 and restore last known-good configuration backup')
        calls=[call('create_change_request',body=body)]+[call('create_change_request_approval',change_request_id=last('change_requests'),body={'approver_email':mail}) for mail in ['network_manager@acme.com','it_director@acme.com']]
        values={k:v for k,v in body.items() if not k.startswith('implementation_')}
        rules=[row_rule('change_requests',values,new=True),Rule('implementation window and two approvals',"SELECT datetime(implementation_window_start)='2025-02-14 23:00:00' AND datetime(implementation_window_end)='2025-02-15 02:00:00' AND (SELECT count(DISTINCT approver_email) FROM change_request_approvals WHERE change_request_id=c.id AND approver_email IN ('network_manager@acme.com','it_director@acme.com'))=2 FROM change_requests c WHERE id=(SELECT max(id) FROM change_requests)")]
        return TaskSpec(calls,rules,[mutate(0,'body.risk_level','low'),{'kind':'omit','index':2,'name':'missing_director_approval'}])
    if i==6:
        where="category='hardware' AND asset_criticality='mission_critical' AND impact_level='high'"
        existing=query(db,'SELECT id FROM sla_priority_matrix WHERE '+where)
        values=dict(category='hardware',asset_criticality='mission_critical',impact_level='high',priority_code='P1',response_time_minutes=60,resolution_time_minutes=240,is_active=1)
        c=call('update_sla_priority_rule',rule_id=existing[0]['id'],body={'priority_code':'P1','response_time_minutes':60,'resolution_time_minutes':240,'is_active':True}) if len(existing)==1 else call('create_sla_priority_rule',body=values)
        return TaskSpec([c],[row_rule('sla_priority_matrix',values)],[mutate(0,'body.response_time_minutes',120),mutate(0,'body.priority_code','P2')])
    if i==7:
        return TaskSpec([],[],status='needs_review',notes='资产故障率的分母、去重和重复故障口径未定义；最近完整季度按运行日为2026Q3，不使用初态2024Q4导出作为答案。',evidence={'requested_locations':['New York HQ','London Office','Singapore Hub'],'undefined_metric':'asset_failure_rates'})
    if i==8:
        values=dict(name='Audit Berlin disk alert',alert_type='disk_space_critical',location_id=loc('Berlin Datacenter'),asset_type='server',incident_category='hardware',impact_level='high',short_description_template='Disk space critical on server {asset_serial_number}',is_active=1)
        return TaskSpec([call('create_monitoring_rule',body={**values,'is_active':True})],[row_rule('monitoring_rules',values,new=True)],[mutate(0,'body.alert_type','disk_space_warning'),mutate(0,'body.impact_level','low')])
    if i==9:
        return TaskSpec([],[],status='needs_review',notes='technician_utilization百分比需要可用工时分母；数据库无排班或工作日历，用户未指定每天可用工时，不能把工具默认8小时作为独立真值。',evidence={'team':'Field Support – West Region','window_days':30})
    raise ValueError(i)
