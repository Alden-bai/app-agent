"""Independent requirements transcribed from the ten original user tasks."""
import json
from datetime import date, datetime, timezone

from awm_audit.common import query
from awm_audit.oracle import (NeedsReview, Rule, TaskSpec, call, mutate, one,
                              resultref, row_rule, scalar, sqlref)


def build(idx, db):
    if idx == 0:
        rooftops = query(db, "SELECT id,organization_id,name FROM dealerships WHERE primary_rooftop_flag=1")
        if len(rooftops) != 1:
            return TaskSpec([], [], status='needs_review', notes='原始数据有多个组织的primary rooftop，当前用户profile只写admin，未指定当前组织，无法唯一确定“my primary dealership”。', evidence={'primary_rooftops':rooftops,'existing_vin':query(db,"SELECT id,dealership_id FROM vehicles WHERE vin='1HGCM82633A123456'")})
        roof = rooftops[0]['id']
        values = dict(dealership_id=roof, vin="1HGCM82633A123456", year=2022, make="Honda", model="Civic",
                      trim="EX", mileage=12500, fuel_type="gasoline", drivetrain="FWD", exterior_color="Blue",
                      features_json=["sunroof", "backup camera", "heated seats"], lot_location="Main Lot A",
                      list_price=23500, discount_amount=500, incentive_amount=750, status="on_lot")
        exists = query(db, "SELECT id,dealership_id,vin FROM vehicles WHERE dealership_id=? AND vin=?", (roof, values['vin']))
        body = {k: v for k, v in values.items() if k != 'dealership_id'}
        body['model_name'] = body.pop('model'); body['features'] = body.pop('features_json');body['id']='AUDIT-HONDA-CIVIC'
        return TaskSpec([call('create_vehicle', dealership_id=roof, body=body)], [row_rule('vehicles', values, new=True)],
                        [mutate(0,'body.mileage',13000),mutate(0,'body.list_price',24000)],
                        status='confirmed_defect' if exists else 'ready',
                        notes='请求创建的新VIN在主店初态已存在；UNIQUE(dealership_id,vin)禁止再创建，不能把已有车辆当作新建成功。' if exists else '',
                        evidence={'existing':exists,'constraint':'UNIQUE(dealership_id, vin)'})
    if idx == 1:
        v=one(db,"SELECT * FROM vehicles WHERE stock_id='FORD-F150-987'")
        calls=[call('get_vehicle_by_stock_id',stock_id='FORD-F150-987'),
               call('update_vehicle_status',vehicle_id=v['id'],body={'status':'on_lot','lot_location':'Truck Row 3'}),
               call('update_vehicle_pricing',vehicle_id=v['id'],body={'list_price':48990}),
               call('update_vehicle_features',vehicle_id=v['id'],body={'mode':'append','features':['tow package']})]
        rules=[row_rule('vehicles',dict(id=v['id'],status='on_lot',lot_location='Truck Row 3',list_price=48990)),
               Rule('add tow package and preserve prior features',"SELECT EXISTS(SELECT 1 FROM json_each(features_json) WHERE value='tow package') AND NOT EXISTS(SELECT value FROM json_each(?) EXCEPT SELECT value FROM json_each(features_json)) FROM vehicles WHERE id=?",[v['features_json'],v['id']])]
        return TaskSpec(calls,rules,[mutate(1,'body.lot_location','Truck Row 2'),mutate(2,'body.list_price',48000)])
    if idx == 2:
        expected=query(db,"SELECT id,make,model,year,list_price,status FROM vehicles WHERE status='in_transit' AND make='Toyota' AND model='RAV4' AND year BETWEEN 2021 AND 2024 AND list_price<38000 ORDER BY expected_arrival_date,id")
        return TaskSpec([call('search_vehicles',status='in_transit',make='Toyota',model='RAV4',min_year=2021,max_year=2024,max_list_price=37999.99,sort_by='expected_arrival_date',sort_direction='asc',limit=10000)],
                        [],[{'kind':'answer','value':'[]','name':'empty_answer'},{'kind':'answer','value':'[{"id":"wrong"}]','name':'wrong_answer'}],
                        answer=resultref(),answer_expected=expected,answer_path=['vehicles'],answer_columns=list(expected[0]) if expected else ['id'])
    if idx == 3:
        d=one(db,"SELECT id FROM dealerships WHERE name LIKE '%Metro Auto%'")['id']
        body=dict(name='New Year SUV Margin Boost',promotion_tag='New Year SUV Margin Boost',start_date='2025-01-01',end_date='2025-03-31',scope='dealership',body_style='SUV',model_name=None,condition_custom_json={'models':['CR-V','Pilot']},rule_type='markup',percentage_adjustment=3,applies_to_specific_vehicles_flag=0)
        values={k:v for k,v in body.items() if k!='model_name'};values['dealership_id']=d
        return TaskSpec([call('create_pricing_rule_for_dealership',dealership_id=d,body=body)],
                        [row_rule('pricing_rules',values,new=True)], [mutate(0,'body.percentage_adjustment',5),mutate(0,'body.body_style','Sedan')])
    if idx == 4:
        d=one(db,"SELECT id FROM dealerships WHERE name LIKE '%Downtown%'")['id']
        body=dict(id='audit-lead-maria',source_channel='website',source_detail='web_form',contact_name='Maria Lopez',contact_email='maria.lopez@example.com',contact_phone='555-0199',interest_level='high',status='open',preferred_vehicle_ids=['STOCK-TOY-CAM-101','STOCK-TOY-CAM-102'],assigned_salesperson_id='sp-204')
        values={k:v for k,v in body.items() if k!='preferred_vehicle_ids'};values.update(dealership_id=d,preferred_vehicle_ids_json=body['preferred_vehicle_ids'])
        return TaskSpec([call('create_lead',dealership_id=d,body=body)], [row_rule('leads',values,new=True)],
                        [mutate(0,'body.assigned_salesperson_id','sp-300'),mutate(0,'body.interest_level','low')])
    if idx == 5:
        attempts=[dict(contact_channel='phone',contact_time='2025-01-05T10:30:00-05:00',notes='left voicemail'),dict(contact_channel='email',contact_time='2025-01-06T16:45:00-05:00',notes='sent follow-up quote')]
        calls=[call('create_contact_attempts_for_lead',lead_id='lead-8831',body={'contact_attempts':attempts}),call('update_lead_interest_level',lead_id='lead-8831',body={'interest_level':'medium'})]
        rules=[row_rule('leads',{'id':'lead-8831','interest_level':'medium'})]
        for a in attempts:
            rules.append(Rule('contact method, instant and notes',"SELECT count(*)=1 FROM contact_attempts t WHERE lead_id='lead-8831' AND contact_channel=? AND datetime(contact_time)=datetime(?) AND notes=? AND NOT EXISTS(SELECT 1 FROM initial.contact_attempts i WHERE i.id=t.id)",[a['contact_channel'],a['contact_time'],a['notes']]))
        for a in attempts:
            a['contact_time']=datetime.fromisoformat(a['contact_time']).astimezone(timezone.utc).isoformat()
        return TaskSpec(calls,rules,[mutate(1,'body.interest_level','low'),mutate(0,['body','contact_attempts',1,'notes'],'wrong')],notes='偏移时区按绝对时刻核对，不把丢失时区的墙上时间自动当作正确。')
    if idx == 6:
        body=dict(lead_id='lead-9027',salesperson_id='sp-112',vehicle_id='STOCK-SUB-OUT-220',appointment_time='2025-02-10T14:00:00-05:00',appointment_location='Subaru East lot',notes='Customer wants to compare Limited vs Premium trims',status='scheduled')
        values={k:v for k,v in body.items() if k!='appointment_time'}
        body['appointment_time']=datetime.fromisoformat(body['appointment_time']).astimezone(timezone.utc).isoformat()
        return TaskSpec([call('create_test_drive_appointment',body=body)], [row_rule('test_drive_appointments',values,new=True),Rule('appointment absolute instant',"SELECT count(*)=1 FROM test_drive_appointments t WHERE lead_id='lead-9027' AND datetime(appointment_time)=datetime(?) AND NOT EXISTS(SELECT 1 FROM initial.test_drive_appointments i WHERE i.id=t.id)",[body['appointment_time']])],
                        [mutate(0,'body.vehicle_id','STOCK-HON-CRV-330'),mutate(0,'body.appointment_location','Wrong lot')])
    if idx == 7:
        body=dict(vehicle_id='STOCK-HON-CRV-330',trade_in_vin='1FTFW1ET1EKF12345',trade_in_estimated_value=8500,final_price=31750,financing_type='dealer_finance',delivery_date='2025-01-20')
        return TaskSpec([call('convert_lead_to_sales_order',lead_id='lead-7612',body=body)],
                        [row_rule('sales_orders',{**body,'lead_id':'lead-7612'},new=True),row_rule('vehicles',{'id':body['vehicle_id'],'status':'sold'})],
                        [mutate(0,'body.final_price',32750),mutate(0,'body.trade_in_estimated_value',7500)])
    if idx == 8:
        return TaskSpec([],[],status='needs_review',notes='inventory_turnover 与 gross_profit_estimates 的业务口径未由用户任务唯一指定（平均库存/期末库存，成本缺失处理等）；不能用官方报表实现自身定义来证明其正确。',evidence={'date_range':['2024-07-01','2024-12-31'],'required':['inventory_turnover','lead_conversion_rates','gross_profit_estimates']})
    if idx == 9:
        d=one(db,"SELECT id FROM dealerships WHERE name LIKE '%Highway Motors%'")['id']
        ids=[x['id'] for x in query(db,"SELECT id FROM vehicles WHERE dealership_id=? AND status='on_lot' AND julianday('now')-julianday(status_on_lot_at)>90 AND list_price>40000 ORDER BY id",(d,))]
        body=dict(name='Aging Highline Clearance',start_date='2025-01-15',end_date='2025-02-15',min_days_on_lot=90,min_list_price=40000,flat_discount_amount=2000)
        rules=[row_rule('pricing_rules',dict(dealership_id=d,name=body['name'],start_date=body['start_date'],end_date=body['end_date'],rule_type='discount',flat_amount_adjustment=2000),new=True)]
        ruleid="(SELECT max(id) FROM pricing_rules WHERE name='Aging Highline Clearance')"
        rules.append(Rule('exact eligible vehicle links',f"SELECT (SELECT count(DISTINCT vehicle_id) FROM pricing_rule_vehicles WHERE pricing_rule_id={ruleid})=? AND NOT EXISTS(SELECT vehicle_id FROM pricing_rule_vehicles WHERE pricing_rule_id={ruleid} AND vehicle_id NOT IN (SELECT value FROM json_each(?)))",[len(ids),json.dumps(ids)]))
        return TaskSpec([call('find_aging_on_lot_vehicles',dealership_id=d,min_days_on_lot=90,min_list_price=40000),call('create_aging_highline_clearance_rule',dealership_id=d,body=body)],rules,[mutate(1,'body.flat_discount_amount',1000),mutate(1,'body.end_date','2025-02-10')],notes='按实际运行日期查超过90天；不回拨到2025年。')
    raise ValueError(idx)
