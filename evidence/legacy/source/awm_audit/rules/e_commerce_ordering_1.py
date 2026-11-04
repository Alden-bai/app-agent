import json
from awm_audit.common import query
from awm_audit.oracle import Rule,TaskSpec,call,mutate,row_rule,scalar,sqlref,one,resultref


def build(i,db):
    part=lambda n:one(db,'SELECT * FROM parts WHERE part_number=?',(n,))
    warehouse=lambda n:scalar(db,'SELECT id FROM warehouses WHERE code=?',(n,))
    latest=lambda t:sqlref(f'SELECT max(id) FROM {t}')
    if i==0:
        sql="""SELECT DISTINCT p.id,p.part_number,p.short_description,p.list_price,(SELECT coalesce(sum(on_hand),0) FROM stock_records s WHERE s.part_id=p.id) total_on_hand FROM parts p JOIN manufacturers m ON m.id=p.manufacturer_id JOIN product_families pf ON pf.id=p.product_family_id JOIN fitment_applications f ON f.part_id=p.id JOIN fitment_positions fp ON fp.id=f.position_id JOIN vehicles v ON v.id=f.vehicle_id JOIN vehicle_makes vm ON vm.id=v.make_id JOIN vehicle_models md ON md.id=v.model_id JOIN engine_codes ec ON ec.id=v.engine_code_id WHERE p.status='active' AND m.name='ACME Brakes' AND pf.name='Brake Pads' AND vm.name='Honda' AND md.name='Civic' AND v.model_year=2018 AND ec.code='R18A1' AND fp.code='front_left' ORDER BY p.id"""
        expected=[{k:v for k,v in x.items() if k!='id'} for x in query(db,sql)]
        args=dict(make_name='Honda',model_name='Civic',model_year=2018,engine_code='R18A1',position_code='front_left',product_family_name='Brake Pads',manufacturer_name='ACME Brakes')
        vehicle=scalar(db,"SELECT v.id FROM vehicles v JOIN vehicle_makes m ON m.id=v.make_id JOIN vehicle_models md ON md.id=v.model_id JOIN engine_codes e ON e.id=v.engine_code_id WHERE m.name='Honda' AND md.name='Civic' AND v.model_year=2018 AND e.code='R18A1'")
        position=scalar(db,"SELECT id FROM fitment_positions WHERE code='front_left'")
        calls=[call('list_parts_endpoint',manufacturer_name='ACME Brakes',product_family_name='Brake Pads',status='active',limit=10000),call('list_fitment_applications_endpoint',vehicle_id=vehicle),call('list_stock_records_endpoint')]
        return TaskSpec(calls,[],[{'kind':'answer','value':'{}','name':'missing_part_fields'},{'kind':'answer','value':'{"parts":[]}','name':'empty_parts'}],answer={'$compute':'parts_stock','position_id':position},answer_expected=expected,answer_path=['parts'],answer_columns=['part_number','short_description','list_price','total_on_hand'],answer_ordered=False,evidence={'independent_selection_sql':sql})
    if i==1:
        values=dict(part_number='ACM-ALT-2201',manufacturer_id=scalar(db,"SELECT id FROM manufacturers WHERE name='ACME Electrical'"),product_family_id=scalar(db,"SELECT id FROM product_families WHERE name='Alternators'"),short_description='ACME 150A Alternator',unit_of_measure='each',list_price=249.99,cost=165,status='active')
        exists=query(db,"SELECT id,part_number FROM parts WHERE part_number='ACM-ALT-2201'")
        return TaskSpec([call('create_part_endpoint',body=values)],[row_rule('parts',values,new=True)],[mutate(0,'body.list_price',299.99),mutate(0,'body.status','inactive')],status='confirmed_defect' if exists else 'needs_review',notes='请求新建的part_number在原始数据库已存在且唯一，不能将原有配件视为新建成功。' if exists else '需补齐两侧fitment链接后才可评分。',evidence={'existing':exists,'constraint':'parts.part_number UNIQUE'})
    if i==2:
        return TaskSpec([],[],status='needs_review',notes='补货量措辞“150 units on_hand plus any current on_order”有歧义：是目标库存150再加在途，还是库存位置on_hand+on_order补到150。两者不同，不能直接采用官方辅助API公式作为真值。',evidence={'part_number':'BOS-12345','target':150})
    if i==3:
        customer=scalar(db,"SELECT id FROM customers WHERE external_ref='JOB-7752' AND type='repair_shop'")
        parts=[part('NGK-IFR6T11'),part('DEN-ALT2100')]
        calls=[call('create_sales_order_endpoint',body={'order_number':'AUDIT-JOB-7752','customer_id':customer,'reference':'JOB-7752','order_type':'repair_shop','status':'open'})]
        rules=[row_rule('sales_orders',{'customer_id':customer,'reference':'JOB-7752'},new=True)];expected=[]
        for j,(p,qty,discount) in enumerate(zip(parts,[3,1],[5,0])):
            stocks=query(db,'SELECT warehouse_id,on_hand FROM stock_records WHERE part_id=? ORDER BY on_hand DESC,warehouse_id',(p['id'],))
            if not stocks:raise ValueError('No warehouse for requested order line')
            wh=stocks[0];backorder='none' if wh['on_hand']>=qty else 'backordered'
            body=dict(line_number=j+1,part_id=p['id'],warehouse_id=wh['warehouse_id'],quantity=qty,unit_price=p['list_price'],discount_percent=discount,backorder_status=backorder,line_status='requested')
            calls.append(call('create_sales_order_line_endpoint',sales_order_id=latest('sales_orders'),body=body))
            rules.append(row_rule('sales_order_lines',body,where='sales_order_id=(SELECT max(id) FROM sales_orders)',new=True))
            expected.append({'line_number':j+1,'part_id':p['id'],'quantity':qty,'unit_price':p['list_price'],'discount_percent':discount,'backorder_status':backorder})
        calls.append(call('get_sales_order_totals_endpoint',sales_order_id=latest('sales_orders')))
        total=3*parts[0]['list_price']*.95+parts[1]['list_price']
        return TaskSpec(calls,rules,[mutate(1,'body.quantity',2),mutate(1,'body.discount_percent',0)],answer=resultref(),answer_expected={'total_after_discount':total,'lines':expected},answer_subset=True)
    if i==4:
        order=scalar(db,"SELECT id FROM sales_orders WHERE order_number='SO-10458'");lines=query(db,'SELECT * FROM sales_order_lines WHERE sales_order_id=? ORDER BY line_number',(order,));calls=[];rules=[]
        for l in lines:
            stock=scalar(db,'SELECT on_hand FROM stock_records WHERE part_id=? AND warehouse_id=?',(l['part_id'],l['warehouse_id']))
            if stock==0:
                rules.append(row_rule('sales_order_lines',{'id':l['id'],'line_status':'requested','backorder_status':'backordered'}))
            else:
                for status in ['confirmed','picked','shipped','delivered']:calls.append(call('update_sales_order_line_status_endpoint',sales_order_line_id=l['id'],body={'line_status':status,'backorder_status':'none','set_timestamp':True}))
                rules += [row_rule('sales_order_lines',{'id':l['id'],'line_status':'delivered'}),Rule('all lifecycle timestamps',"SELECT confirmed_at IS NOT NULL AND picked_at IS NOT NULL AND shipped_at IS NOT NULL AND delivered_at IS NOT NULL AND julianday(confirmed_at)<=julianday(picked_at) AND julianday(picked_at)<=julianday(shipped_at) AND julianday(shipped_at)<=julianday(delivered_at) FROM sales_order_lines WHERE id=?",[l['id']])]
        return TaskSpec(calls,rules,[mutate(len(calls)-1,'body.line_status','shipped'),{'kind':'omit','index':max(0,len(calls)-2),'name':'lifecycle_incomplete'}])
    if i==5:
        eligible=query(db,"SELECT p.id part_id,s.warehouse_id FROM parts p JOIN product_families f ON f.id=p.product_family_id JOIN stock_records s ON s.part_id=p.id WHERE p.status='active' AND f.name='Radiators' AND julianday('now')-julianday(p.created_at)>365 AND s.on_hand>50 AND NOT EXISTS(SELECT 1 FROM sales_order_lines l WHERE l.part_id=p.id AND datetime(l.requested_at)>=datetime('now','-180 days')) ORDER BY p.id,s.warehouse_id")
        ids=sorted({r['part_id'] for r in eligible})
        if not ids:return TaskSpec([],[],status='needs_review',notes='无满足老化条件的配件，当前初态不能建立有效操作正反例。')
        calls=[call('bulk_mark_parts_superseded_endpoint',body={'part_ids':ids}),call('create_aging_inventory_report_entries_endpoint',body={'entries':[{**r,'reason':'Aging inventory: over 365 days, no sales in 180 days'} for r in eligible]})]
        rules=[row_rule('parts',{'id':p,'status':'superseded'}) for p in ids]+[row_rule('aging_inventory_reports',r,new=True) for r in eligible]
        rules.append(Rule('no unrelated parts superseded',"SELECT NOT EXISTS(SELECT 1 FROM parts p JOIN initial.parts i ON i.id=p.id WHERE p.status IS NOT i.status AND p.id NOT IN (SELECT value FROM json_each(?)))",[json.dumps(ids)]))
        return TaskSpec(calls,rules,[{'kind':'omit','index':1,'name':'no_aging_report'},mutate(0,'body.part_ids',ids[:-1])],evidence={'eligible':eligible})
    if i==6:
        return TaskSpec([],[],status='needs_review',notes='数据库没有首次发货尝试及历史缺货事件；当前backorder_status不能独立证明“shipped on first attempt”。不能照搬官方报表用当前非缺货行数代替历史首次发货量。',evidence={'metric':'fill_rate','period':'last full calendar month','alert_threshold_percent':92})
    if i==7:
        expected=[]
        for p in query(db,"SELECT p.id,p.part_number,sum(l.quantity_ordered) total_ordered_quantity,sum(l.quantity_ordered*l.unit_cost) total_extended_cost FROM purchase_order_lines l JOIN purchase_orders o ON o.id=l.purchase_order_id JOIN supplier_accounts a ON a.id=o.supplier_account_id JOIN parts p ON p.id=l.part_id WHERE a.name='Global Bearings Co' AND o.status='open' GROUP BY p.id ORDER BY total_extended_cost DESC,p.id"):
            details=query(db,'SELECT w.id warehouse_id,w.code warehouse_code,s.aisle,s.bin_location FROM stock_records s JOIN warehouses w ON w.id=s.warehouse_id WHERE s.part_id=? ORDER BY w.id',(p['id'],))
            expected.append({'part_number':p['part_number'],'part_id':p['id'],'total_ordered_quantity':p['total_ordered_quantity'],'total_extended_cost':p['total_extended_cost'],'warehouse_details':details})
        answer={'$result':-1,'$sort':{'path':['parts'],'by':'total_extended_cost','reverse':True}}
        return TaskSpec([call('get_supplier_open_po_summary_endpoint',supplier_name='Global Bearings Co')],[],[{'kind':'answer','value':'{}','name':'missing_locations'},{'kind':'answer','value':'{"parts":[]}','name':'empty_summary'}],answer=answer,answer_expected=expected,answer_path=['parts'])
    if i==8:
        s=one(db,"SELECT s.* FROM stock_records s JOIN parts p ON p.id=s.part_id JOIN warehouses w ON w.id=s.warehouse_id WHERE p.part_number='MON-SHOCK-9900' AND w.code='ATL-01'")
        values=dict(id=s['id'],on_hand=s['on_hand']+25,on_order=s['on_order']-25)
        rules=[row_rule('stock_records',values)]
        if values['on_hand']-s['reorder_point']>40:rules.append(Rule('potential overstock note',"SELECT lower(coalesce(internal_note,'')) LIKE '%overstock%' FROM stock_records WHERE id=?",[s['id']]))
        calls=[call('action_receive_stock_and_flag_overstock_endpoint',body={'part_number':'MON-SHOCK-9900','warehouse_code':'ATL-01','received_quantity':25,'overstock_margin':40})]
        return TaskSpec(calls,rules,[mutate(0,'body.received_quantity',20),mutate(0,'body.warehouse_code','CHI-02')])
    if i==9:
        p=one(db,"SELECT DISTINCT p.* FROM parts p JOIN product_families f ON f.id=p.product_family_id JOIN fitment_applications a ON a.part_id=p.id JOIN vehicles v ON v.id=a.vehicle_id JOIN vehicle_makes m ON m.id=v.make_id JOIN vehicle_models md ON md.id=v.model_id JOIN engine_codes e ON e.id=v.engine_code_id WHERE p.status='active' AND f.name='Air Filters' AND m.name='Toyota' AND md.name='Camry' AND v.model_year=2020 AND e.code='A25A-FKS' ORDER BY p.list_price,p.id LIMIT 1")
        wh=warehouse('CHI-02');stock=scalar(db,'SELECT on_hand FROM stock_records WHERE part_id=? AND warehouse_id=?',(p['id'],wh));customer=scalar(db,"SELECT id FROM customers WHERE type='counter_sale' ORDER BY id LIMIT 1")
        body=dict(line_number=1,part_id=p['id'],warehouse_id=wh,quantity=1,unit_price=p['list_price'],discount_percent=0,line_status='requested',backorder_status='none' if stock>=1 else 'backordered')
        calls=[call('list_parts_endpoint',product_family_name='Air Filters',status='active',limit=10000),call('create_sales_order_endpoint',body={'order_number':'AUDIT-COUNTER-CAMRY','customer_id':customer,'order_type':'counter_sale','status':'open'}),call('create_sales_order_line_endpoint',sales_order_id=latest('sales_orders'),body=body)]
        if stock>=1:
            calls += [call('update_sales_order_line_status_endpoint',sales_order_line_id=latest('sales_order_lines'),body={'line_status':s,'backorder_status':'none','set_timestamp':True}) for s in ['picked','shipped']]
        rules=[row_rule('sales_orders',{'customer_id':customer,'order_type':'counter_sale'},new=True),row_rule('sales_order_lines',{**body,'line_status':'shipped' if stock>=1 else 'requested'},where='sales_order_id=(SELECT max(id) FROM sales_orders)',new=True),Rule('exactly one counter-sale line',"SELECT count(*)=1 FROM sales_order_lines WHERE sales_order_id=(SELECT max(id) FROM sales_orders)")]
        return TaskSpec(calls,rules,[mutate(2,'body.quantity',2),{'kind':'omit','index':2,'name':'no_order_line'}],evidence={'selected_cheapest_part':p,'initial_on_hand':stock})
    raise ValueError(i)
