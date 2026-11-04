import json
from awm_audit.common import query
from awm_audit.oracle import Rule,TaskSpec,call,mutate,row_rule,scalar,sqlref,one,resultref


VEGAN="""(EXISTS(SELECT 1 FROM sku_dietary_tags x JOIN dietary_tags d ON d.id=x.dietary_tag_id WHERE x.sku_id=s.id AND lower(d.code)='vegan') OR EXISTS(SELECT 1 FROM product_dietary_tags x JOIN dietary_tags d ON d.id=x.dietary_tag_id WHERE x.product_id=s.product_id AND lower(d.code)='vegan'))"""
NO_ALLERGENS="""NOT EXISTS(SELECT a.name FROM sku_allergens x JOIN allergens a ON a.id=x.allergen_id WHERE x.sku_id=s.id AND lower(a.name) IN (SELECT lower(value) FROM json_each(?))) AND NOT EXISTS(SELECT a.name FROM product_allergens x JOIN allergens a ON a.id=x.allergen_id WHERE x.product_id=s.product_id AND lower(a.name) IN (SELECT lower(value) FROM json_each(?))) AND NOT EXISTS(SELECT a.name FROM sku_ingredients x JOIN ingredient_allergens ia ON ia.ingredient_id=x.ingredient_id JOIN allergens a ON a.id=ia.allergen_id WHERE x.sku_id=s.id AND lower(a.name) IN (SELECT lower(value) FROM json_each(?)))"""


def build(i,db):
    cart=scalar(db,"SELECT id FROM carts WHERE user_id=1 AND status='open'")
    latest=lambda t:sqlref(f'SELECT max(id) FROM {t}')
    if i==0:
        banned=json.dumps(['soy','lactose'])
        sql=f"SELECT s.id,s.base_price FROM sku s JOIN products p ON p.id=s.product_id JOIN product_categories c ON c.id=p.category_id WHERE lower(c.code)='protein_powder' AND lower(s.flavor)='vanilla' AND {VEGAN} AND {NO_ALLERGENS} AND EXISTS(SELECT 1 FROM product_goals pg JOIN goals g ON g.id=pg.goal_id WHERE pg.product_id=p.id AND lower(g.name)='muscle gain') ORDER BY s.base_price,s.id LIMIT 2"
        skus=query(db,sql,[banned]*3)
        if len(skus)!=2:return TaskSpec([],[],status='confirmed_defect',notes='初态不足两款同时满足全部条件的香草SKU。',evidence={'eligible_skus':skus})
        calls=[call('search_skus',product_category_code='PROTEIN_POWDER',flavor='vanilla')]+[call('add_item_to_current_cart',body={'sku_id':s['id'],'quantity':2,'purchase_type':'one_time','unit_price':s['base_price']}) for s in skus]
        rules=[]
        for s in skus:
            old=scalar(db,'SELECT coalesce(sum(quantity),0) FROM cart_items WHERE cart_id=? AND sku_id=? AND purchase_type=\'one_time\'',(cart,s['id']))
            rules.append(Rule('add two of chosen cheapest compliant SKU',"SELECT coalesce(sum(quantity),0)=? FROM cart_items WHERE cart_id=? AND sku_id=? AND purchase_type='one_time'",[old+2,cart,s['id']]))
        return TaskSpec(calls,rules,[mutate(1,'body.quantity',1),{'kind':'omit','index':2,'name':'missing_second_sku'}],evidence={'independent_selection_sql':sql,'selected':skus})
    if i==1:
        sku=one(db,"SELECT id FROM sku WHERE sku_code='PRE-ENERGY-300GFRT'")
        address=scalar(db,'SELECT id FROM addresses WHERE user_id=1 AND is_default_shipping=1')
        bad=query(db,"SELECT i.name FROM sku_ingredients x JOIN ingredients i ON i.id=x.ingredient_id WHERE x.sku_id=? AND (lower(i.name) LIKE '%sucralose%' OR lower(i.name) LIKE '%aspartame%' OR lower(i.name) LIKE '%acesulfame%')",(sku['id'],))
        body=dict(sku_code='PRE-ENERGY-300GFRT',interval_unit='week',interval_count=4,use_default_shipping_address=True,require_no_artificial_sweeteners=True)
        return TaskSpec([call('get_user_preferences'),call('get_sku_by_code',sku_code=body['sku_code']),call('create_subscription',body=body)],[row_rule('subscriptions',{'user_id':1,'sku_id':sku['id'],'interval_unit':'week','interval_count':4,'shipping_address_id':address,'require_no_artificial_sweeteners':1,'status':'active'},new=True),Rule('chosen SKU ingredient compliance',"SELECT ?",[int(not bad)])],[mutate(2,'body.interval_count',2),mutate(2,'body.require_no_artificial_sweeteners',False)],status='confirmed_defect' if bad else 'ready',notes='请求SKU及其同产品全部口味初态含人工甜味剂，不能同时创建该产品订阅并满足用户指定的仅合规口味条件。' if bad else '',evidence={'artificial_sweeteners_found':bad,'all_flavors':query(db,"SELECT s.sku_code,i.name ingredient FROM sku s JOIN sku_ingredients x ON x.sku_id=s.id JOIN ingredients i ON i.id=x.ingredient_id WHERE s.product_id=(SELECT product_id FROM sku WHERE id=?) ORDER BY s.id,i.id",(sku['id'],))})
    if i==2:
        bundle=one(db,"SELECT b.id FROM bundles b WHERE EXISTS(SELECT 1 FROM product_training_topics pt JOIN training_topics t ON t.id=pt.training_topic_id WHERE pt.product_id=b.product_id AND lower(t.name)='endurance') AND EXISTS(SELECT 1 FROM bundle_items bi JOIN sku s ON s.id=bi.sku_id WHERE bi.bundle_id=b.id AND s.is_electrolyte_mix=1) AND EXISTS(SELECT 1 FROM bundle_items bi JOIN sku s ON s.id=bi.sku_id WHERE bi.bundle_id=b.id AND s.is_bcaa_capsules=1)")['id']
        old=scalar(db,'SELECT coalesce(sum(quantity),0) FROM cart_items WHERE cart_id=? AND bundle_id=?',(cart,bundle))
        calls=[call('list_bundles',training_topic_name='endurance'),call('add_item_to_current_cart',body={'bundle_id':bundle,'quantity':1,'purchase_type':'one_time'}),call('remove_single_item_electrolytes_from_cart',body={'remove_electrolyte_single_items':True})]
        other=scalar(db,'SELECT id FROM bundles WHERE id!=? ORDER BY id LIMIT 1',(bundle,))
        return TaskSpec(calls,[Rule('selected bundle added',"SELECT coalesce(sum(quantity),0)>=? FROM cart_items WHERE cart_id=? AND bundle_id=?",[old+1,cart,bundle]),Rule('no single electrolyte items remain',"SELECT count(*)=0 FROM cart_items ci JOIN sku s ON s.id=ci.sku_id WHERE ci.cart_id=? AND ci.bundle_id IS NULL AND s.is_electrolyte_mix=1",[cart])],[mutate(1,'body.bundle_id',other),{'kind':'omit','index':2,'name':'electrolytes_not_removed'}],notes='原任务未明确数量上限，不能将添加2份直接标为错误；仅检查确实添加目标套餐，并清除独立电解质商品。原数量变体移出负例统计，新增错误套餐变体须实际执行后才可计分。')
    if i==3:
        order=scalar(db,"SELECT id FROM orders WHERE user_id=1 AND status='open'")
        items=query(db,'SELECT id,sku_id,quantity FROM order_items WHERE order_id=?',(order,));short=[];rules=[]
        for item in items:
            lots=query(db,"SELECT l.id,l.quantity_on_hand FROM inventory_lots l JOIN warehouses w ON w.id=l.warehouse_id WHERE l.sku_id=? AND w.code='WH-EAST-01' AND (l.expiration_date IS NULL OR date(l.expiration_date)>=date('now')) AND l.quantity_on_hand>0 ORDER BY coalesce(l.expiration_date,'9999-12-31'),l.id",(item['sku_id'],))
            remaining=item['quantity'];alloc={}
            for lot in lots:
                n=min(remaining,lot['quantity_on_hand'])
                if n:alloc[lot['id']]=n;remaining-=n
            if remaining:short.append({'order_item':item,'missing_quantity':remaining,'valid_lots':lots})
            for lid,n in alloc.items():rules.append(Rule('FEFO exact allocation',"SELECT coalesce(sum(quantity),0)=? FROM order_item_lots WHERE order_item_id=? AND inventory_lot_id=? AND id NOT IN (SELECT id FROM initial.order_item_lots)",[n,item['id'],lid]))
        if short:return TaskSpec([],[],status='confirmed_defect',notes='原始未过期库存不足，不能在不改数据的情况下按FEFO完整配货并履约。',evidence={'shortages':short,'warehouse':'WH-EAST-01'})
        calls=[call('simulate_fefo_allocation_for_order',order_id=order,body={'warehouse_code':'WH-EAST-01','commit_allocation':True}),call('update_order_status',order_id=order,body={'status':'fulfilled'})]
        rules.append(row_rule('orders',{'id':order,'status':'fulfilled'}))
        return TaskSpec(calls,rules,[mutate(0,'body.commit_allocation',False),{'kind':'omit','index':1,'name':'not_fulfilled'}])
    if i==4:
        old=[x['name'] for x in query(db,'SELECT a.name FROM user_allergen_filters f JOIN allergens a ON a.id=f.allergen_id WHERE f.user_id=1')];names=sorted(set(old+['tree nuts','shellfish']));encoded=json.dumps(names)
        rows=query(db,f"SELECT s.id,s.sku_code,s.name,s.flavor,s.base_price FROM sku s JOIN products p ON p.id=s.product_id JOIN product_categories c ON c.id=p.category_id WHERE lower(c.code)='protein_bar' AND {VEGAN} AND {NO_ALLERGENS} AND EXISTS(SELECT 1 FROM inventory_lots l JOIN warehouses w ON w.id=l.warehouse_id WHERE l.sku_id=s.id AND w.code='WH-WEST-02' AND l.quantity_on_hand>0) ORDER BY s.id",[encoded]*3)
        calls=[call('update_user_preferences',body={'require_vegan':True}),call('set_user_allergen_filters',body={'allergen_names':names}),call('list_compliant_protein_bar_skus',warehouse_code='WH-WEST-02')]
        rules=[row_rule('user_preferences',{'user_id':1,'require_vegan':1}),Rule('all requested and preexisting allergen filters retained',"SELECT count(DISTINCT a.name)=? FROM user_allergen_filters f JOIN allergens a ON a.id=f.allergen_id WHERE f.user_id=1 AND a.name IN (SELECT value FROM json_each(?))",[len(names),encoded])]
        # The convenience route compares uppercase VEGAN with lowercase catalog tags.
        # Fetch each candidate's ingredient/allergen/stock detail and assemble the answer.
        candidates=query(db,"SELECT s.sku_code FROM sku s JOIN products p ON p.id=s.product_id JOIN product_categories c ON c.id=p.category_id WHERE c.code='PROTEIN_BAR' ORDER BY s.id")
        calls=calls[:2]+[call('get_sku_by_code',sku_code=r['sku_code']) for r in candidates]
        no_art=bool(scalar(db,'SELECT require_no_artificial_sweeteners FROM user_preferences WHERE user_id=1'))
        return TaskSpec(calls,rules,[mutate(0,'body.require_vegan',False),{'kind':'answer','name':'wrong_sku_list','value':'{"skus":[]}'}],answer={'$compute':'protein_bars','indices':list(range(2,len(calls))),'excluded':names,'warehouse':'WH-WEST-02','no_artificial_sweeteners':no_art},answer_expected=[{'sku_id':r['id'],'sku_code':r['sku_code']} for r in rows],answer_path=['skus'],answer_columns=['sku_id','sku_code'])
    if i==5:
        body=dict(code='audit-recall-2209A',batch_id='BCAA-LEUCINE-LOT-2209A',description='Simulated recall of ingredient batch',starts_at=sqlref("SELECT strftime('%Y-%m-%dT%H:%M:%S','now')"))
        affected=[x['id'] for x in query(db,"SELECT DISTINCT o.id FROM orders o JOIN order_items oi ON oi.order_id=o.id JOIN order_item_lots oil ON oil.order_item_id=oi.id JOIN inventory_lots l ON l.id=oil.inventory_lot_id WHERE l.batch_id='BCAA-LEUCINE-LOT-2209A' AND o.status IN ('open','fulfilled') AND datetime(o.created_at)>=datetime('now','-60 days') ORDER BY o.id")]
        calls=[call('list_inventory_lots',batch_id=body['batch_id']),call('create_recall_event',body=body),call('list_affected_orders_for_recall',recall_event_id=latest('recall_events'),days_back=60),call('tag_orders_for_recall_event',recall_event_id=latest('recall_events'),body={'flag_code':'recall-2209A','order_ids':affected})]
        rules=[row_rule('recall_events',{k:v for k,v in body.items() if k!='starts_at'},new=True),Rule('exact affected-order flags',"SELECT count(DISTINCT order_id)=? AND NOT EXISTS(SELECT 1 FROM order_recall_flags f WHERE f.recall_event_id=(SELECT max(id) FROM recall_events) AND f.order_id NOT IN (SELECT value FROM json_each(?))) FROM order_recall_flags WHERE recall_event_id=(SELECT max(id) FROM recall_events) AND flag_code='recall-2209A'",[len(affected),json.dumps(affected)])]
        return TaskSpec(calls,rules,[mutate(1,'body.batch_id','wrong-batch'),{'kind':'omit','index':2,'name':'no_affected_order_query'}],required_tools=['list_inventory_lots','list_affected_orders_for_recall','tag_orders_for_recall_event'],evidence={'affected_order_ids':affected})
    if i==6:
        sid=scalar(db,"SELECT id FROM subscriptions WHERE external_code='SUB-OMEGA3-90DAY' AND user_id=1")
        calls=[call('pause_subscription_cycles',subscription_external_code='SUB-OMEGA3-90DAY',body={'start_cycle':1,'end_cycle':2,'reason':'too many deliveries'}),call('schedule_subscription_action',subscription_external_code='SUB-OMEGA3-90DAY',body={'action_type':'email_offer','execute_after':sqlref("SELECT strftime('%Y-%m-%dT%H:%M:%S','now','+45 days')"),'metadata':json.dumps({'discount_percent':15,'condition':'not_reactivated'})})]
        rules=[row_rule('subscriptions',{'id':sid,'status':'paused'}),row_rule('subscription_pauses',{'subscription_id':sid,'start_cycle':1,'end_cycle':2,'reason':'too many deliveries'},new=True),Rule('conditional 15 percent offer after 45 days',"SELECT count(*)=1 FROM subscription_scheduled_actions t WHERE subscription_id=? AND action_type='email_offer' AND abs(julianday(execute_after)-julianday(created_at)-45)<0.0001 AND json_extract(metadata,'$.discount_percent')=15 AND json_extract(metadata,'$.condition')='not_reactivated' AND id NOT IN (SELECT id FROM initial.subscription_scheduled_actions)",[sid])]
        # Pausing future cycles does not require changing the immediate global status.
        rules=rules[1:]
        return TaskSpec(calls,rules,[mutate(0,'body.end_cycle',3),mutate(1,'body.metadata',json.dumps({'discount_percent':5,'condition':'not_reactivated'}))])
    if i==7:
        sku=scalar(db,"SELECT id FROM sku WHERE sku_code='WHEY-ISOLATE-CHOC-2LB'")
        body=dict(sku_code='WHEY-ISOLATE-CHOC-2LB',rating=5,review_text='Great taste, mixes easily, and no stomach issues even with my lactose sensitivity',status='pending')
        return TaskSpec([call('create_review',body=body)],[row_rule('reviews',{'user_id':1,'sku_id':sku,'rating':5,'review_text':body['review_text'],'status':'pending'},new=True)],[mutate(0,'body.rating',4),mutate(0,'body.status','approved')])
    if i==8:
        line=one(db,"SELECT oi.id,oi.order_id,oi.quantity,oi.unit_price FROM order_items oi JOIN orders o ON o.id=oi.order_id JOIN sku s ON s.id=oi.sku_id JOIN products p ON p.id=s.product_id WHERE o.user_id=1 AND o.status='fulfilled' AND lower(p.name) LIKE '%creatine%monohydrate%' ORDER BY o.fulfilled_at DESC,o.id DESC LIMIT 1")
        refund=line['quantity']*line['unit_price'];body=dict(order_id=line['order_id'],status='awaiting inspection',reason='clumped product',estimated_refund=refund)
        return TaskSpec([call('get_order_by_id',order_id=line['order_id']),call('create_rma',body=body),call('add_item_to_rma',rma_id=latest('rmas'),body={'order_item_id':line['id'],'quantity':line['quantity'],'reason':'clumped product'})],[row_rule('rmas',{'user_id':1,**body},new=True),row_rule('rma_items',{'order_item_id':line['id'],'quantity':line['quantity'],'reason':'clumped product'},where='rma_id=(SELECT max(id) FROM rmas)'),Rule('only one returned line',"SELECT count(*)=1 FROM rma_items WHERE rma_id=(SELECT max(id) FROM rmas)")],[mutate(1,'body.estimated_refund',refund+5),{'kind':'omit','index':2,'name':'missing_return_item'}])
    if i==9:
        return TaskSpec([],[],status='needs_review',notes='需单独确定限时活动的起止时间及税费/运费政策；不能把工具内部硬编码税率和全购物车折扣当作独立正确答案。',evidence={'goal':'muscle gain','bundles_only':True,'discount_percent':20,'strict_cart_threshold':150})
    raise ValueError(i)
