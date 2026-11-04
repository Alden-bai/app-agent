import json
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
from awm_audit.common import query
from awm_audit.oracle import Rule,TaskSpec,call,mutate,row_rule,scalar,sqlref,one,resultref


def build(i,db):
    restaurant_ids=query(db,'SELECT DISTINCT s.restaurant_id FROM carts c JOIN storefronts s ON s.id=c.storefront_id JOIN customers u ON u.id=c.customer_id WHERE u.user_id=1 AND c.status=\'active\'')
    if len(restaurant_ids)!=1:return TaskSpec([],[],status='needs_review',notes='无法从当前用户active carts唯一定位工作餐厅。',evidence=restaurant_ids)
    rid=restaurant_ids[0]['restaurant_id'];sf=scalar(db,'SELECT id FROM storefronts WHERE restaurant_id=? AND is_main=1',(rid,))
    customer=scalar(db,'SELECT id FROM customers WHERE user_id=1')
    timezone=scalar(db,'SELECT timezone FROM restaurants WHERE id=?',(rid,));today=datetime.now(ZoneInfo(timezone)).date()
    latest=lambda t:sqlref(f'SELECT max(id) FROM {t}')
    if i==0:
        calls=[call('create_menu',restaurant_id=rid,body={'name':'Winter Specials','service_period':'dinner','is_active':True})]
        menu=latest('menus')
        for j,(name,cents) in enumerate([('Truffle Mushroom Risotto',2400),('Slow-Braised Short Ribs',2900)]):
            calls += [call('create_menu_item',restaurant_id=rid,body={'name':name,'base_price_cents':cents,'is_active':True}),call('attach_menu_item_to_menu',menu_id=menu,body={'menu_item_id':latest('menu_items'),'sort_order':j+1,'is_active':True})]
        calls += [call('create_menu_availability_rule',menu_id=menu,body={'day_of_week':d,'start_time':'17:00','end_time':'22:00'}) for d in [4,5]]
        fronts=[x['id'] for x in query(db,"SELECT id FROM storefronts WHERE restaurant_id=? AND channel IN ('web','qr')",(rid,))]
        calls += [call('link_menu_to_storefront',menu_id=menu,body={'storefront_id':s,'is_active':True}) for s in fronts]
        rules=[row_rule('menus',{'restaurant_id':rid,'name':'Winter Specials','service_period':'dinner'},new=True),Rule('exact two dishes with prices',"SELECT count(*)=2 AND sum((m.name='Truffle Mushroom Risotto' AND m.base_price_cents=2400) OR (m.name='Slow-Braised Short Ribs' AND m.base_price_cents=2900))=2 FROM menu_menu_items l JOIN menu_items m ON m.id=l.menu_item_id WHERE l.menu_id=(SELECT max(id) FROM menus) AND l.is_active=1"),Rule('only Friday Saturday evening availability',"SELECT count(*)=2 AND sum(day_of_week IN (4,5) AND start_time='17:00' AND end_time='22:00')=2 AND count(DISTINCT day_of_week)=2 FROM menu_availability_rules WHERE menu_id=(SELECT max(id) FROM menus)"),Rule('web and QR storefront assignment',"SELECT count(DISTINCT storefront_id)=? FROM menu_storefronts WHERE menu_id=(SELECT max(id) FROM menus) AND is_active=1 AND storefront_id IN (SELECT value FROM json_each(?))",[len(fronts),json.dumps(fronts)])]
        return TaskSpec(calls,rules,[mutate(1,'body.base_price_cents',2500),{'kind':'omit','index':len(calls)-1,'name':'missing_storefront'}],notes='工作餐厅由当前用户active carts唯一关联的餐厅确定；此上下文假设写入规则。')
    if i==1:
        existing=query(db,"SELECT id,day_of_week FROM service_hours WHERE storefront_id=? AND fulfillment_type='pickup'",(sf,))
        calls=[call('update_service_hours',id=r['id'],body={'lead_time_minutes':25,**({'is_active':False} if r['day_of_week']==today.weekday() else {})}) for r in existing]
        start=len(calls)
        for begin,end in [('11:30','15:00'),('17:00','21:00')]:calls.append(call('create_service_hours',storefront_id=sf,body={'fulfillment_type':'pickup','day_of_week':today.weekday(),'start_time':begin,'end_time':end,'lead_time_minutes':25,'is_active':True}))
        rules=[Rule('only two pickup windows for local today',"SELECT count(*)=2 AND sum((start_time='11:30' AND end_time='15:00') OR (start_time='17:00' AND end_time='21:00'))=2 AND min(lead_time_minutes)=25 AND max(lead_time_minutes)=25 FROM service_hours WHERE storefront_id=? AND fulfillment_type='pickup' AND day_of_week=? AND is_active=1",[sf,today.weekday()]),Rule('all main-store pickup lead times',"SELECT count(*)=0 FROM service_hours WHERE storefront_id=? AND fulfillment_type='pickup' AND is_active=1 AND lead_time_minutes!=25",[sf])]
        return TaskSpec(calls,rules,[mutate(start,'body.lead_time_minutes',15),mutate(start+1,'body.end_time','22:00')],evidence={'restaurant_timezone':timezone,'local_today':today.isoformat()})
    if i==2:
        return TaskSpec([],[],status='needs_review',notes='任务未给出favorite restaurant字段或关系；当前用户有购物车但这不证明是最喜欢的餐厅，且原购物车已含请求菜品。需确认上下文和“add one”是增量还是最终数量，不能猜测后算正确。',evidence={'current_active_restaurant_id':rid,'local_today':today.isoformat()})
    if i==3:
        cart=one(db,"SELECT * FROM carts WHERE customer_id=? AND status='active' AND fulfillment_type='delivery'",(customer,))
        payment=scalar(db,'SELECT id FROM payment_methods WHERE customer_id=? AND is_default=1',(customer,));address=scalar(db,'SELECT id FROM addresses WHERE customer_id=? ORDER BY id LIMIT 1',(customer,))
        calls=[call('apply_coupon_to_cart',cart_id=cart['id'],body={'coupon_code':'WELCOME10'}),call('get_cart_price_breakdown',cart_id=cart['id']),call('set_cart_tip',cart_id=cart['id'],body={'tip_cents':500}),call('checkout_cart_to_order',cart_id=cart['id'],body={'payment_method_id':payment,'scheduled_time':today.isoformat()+'T19:15:00','delivery_address_id':address})]
        rules=[row_rule('orders',{'customer_id':customer,'cart_id':cart['id'],'fulfillment_type':'delivery','coupon_code':'WELCOME10','tip_cents':500,'payment_method_id':payment},new=True),Rule('ten percent subtotal discount and requested local time',"SELECT abs(discount_cents-subtotal_cents*0.1)<=0.5 AND date(scheduled_time)=? AND time(scheduled_time)='19:15:00' FROM orders WHERE id=(SELECT max(id) FROM orders)",[today.isoformat()]),Rule('pizza and salad present in order',"SELECT count(DISTINCT name)=2 FROM order_items WHERE order_id=(SELECT max(id) FROM orders) AND name IN ('Margherita Pizza','Caesar Salad')")]
        return TaskSpec(calls,rules,[mutate(2,'body.tip_cents',0),{'kind':'omit','index':3,'name':'not_checked_out'}],required_tools=['get_cart_price_breakdown'])
    if i==4:
        orders=query(db,"SELECT id,promised_time FROM orders WHERE restaurant_id=? AND status='placed' AND service_period='dinner' ORDER BY id",(rid,));calls=[];rules=[]
        for o in orders:
            calls.append(call('change_order_status',order_id=o['id'],body={'new_status':'accepted'}));rules.append(row_rule('orders',{'id':o['id'],'status':'accepted'}))
            if o['promised_time'] and o['promised_time'][11:16]<'19:00':
                when=o['promised_time'][:10]+'T19:15:00';note='Adjusted promise time due to high volume.'
                calls.append(call('update_order_promised_time',order_id=o['id'],body={'promised_time':when,'internal_note':note}))
                rules.append(Rule('adjusted promise and internal note',"SELECT datetime(promised_time)=datetime(?) AND (internal_note LIKE ? OR EXISTS(SELECT 1 FROM order_internal_notes n WHERE n.order_id=o.id AND n.note=?)) FROM orders o WHERE id=?",[when,'%'+note+'%',note,o['id']]))
        if not calls:return TaskSpec([],[],status='needs_review',notes='原始数据当前无placed dinner orders，无法构建有效操作正反例。')
        return TaskSpec(calls,rules,[mutate(0,'body.new_status','cancelled'),{'kind':'omit','index':len(calls)-1,'name':'last_update_omitted'}])
    if i==5:
        order=scalar(db,"SELECT id FROM orders WHERE restaurant_id=? AND status='in_kitchen' AND fulfillment_type='pickup' ORDER BY id LIMIT 1",(rid,))
        calls=[call('change_order_status',order_id=order,body={'new_status':'ready'}),call('change_order_status',order_id=order,body={'new_status':'picked_up'}),call('update_order_metadata',order_id=order,body={'meta':json.dumps({'actual_pickup_timestamp':'__runtime__'})})]
        calls[2]['arguments']['body']['meta']={'actual_pickup_timestamp':sqlref("SELECT max(changed_at) FROM order_status_history WHERE order_id=? AND new_status='picked_up'",params=[order])}
        rules=[row_rule('orders',{'id':order,'status':'picked_up'}),Rule('ready then picked_up transitions and pickup metadata',"SELECT datetime(json_extract(meta,'$.actual_pickup_timestamp'))=(SELECT datetime(max(changed_at)) FROM order_status_history h WHERE h.order_id=o.id AND h.new_status='picked_up' AND h.id NOT IN (SELECT id FROM initial.order_status_history)) AND EXISTS(SELECT 1 FROM order_status_history h WHERE h.order_id=o.id AND h.new_status='ready' AND h.id NOT IN (SELECT id FROM initial.order_status_history)) FROM orders o WHERE id=?",[order])]
        # Missing required metadata is a known violation rather than SQL NULL/unknown.
        rules[-1].sql=rules[-1].sql.replace('SELECT datetime(', 'SELECT coalesce(datetime(',1).replace(' FROM orders o WHERE id=?', ',0) FROM orders o WHERE id=?')
        return TaskSpec(calls,rules,[mutate(1,'body.new_status','cancelled'),{'kind':'omit','index':2,'name':'missing_pickup_metadata'}],notes='真实工具调用已复现JSON元数据未持久化；这是该接口缺陷证据，尚不宣称所有替代路径均无解。',evidence={'diagnostic':'postdiagnostics/metadata_persistence.json'})
    if i==6:
        return TaskSpec([],[],status='needs_review',notes='Christmas Day和New Year’s Day未指定年份或是否年度重复；日期表仅支持具体日期，原始订单跨年。先确认年份范围，不能自行决定后称答案正确。',evidence={'restaurant_timezone':timezone,'local_today':today.isoformat()})
    if i==7:
        return TaskSpec([],[],status='needs_review',notes='需建立恰好15单的真实满额时段及API可用性断言；只有写入capacity配置不能证明容量限制生效，且餐厅所有权上下文未明确。',evidence={'fulfillment_type':'delivery','weekday':'Friday','window':['17:00','21:00'],'slot_minutes':30,'capacity':15})
    if i==8:
        expected=query(db,"SELECT lp.customer_id,lp.points_balance,(SELECT count(*) FROM orders o WHERE o.customer_id=lp.customer_id AND o.restaurant_id=lp.restaurant_id AND date(o.created_at)>=date('now','-90 days') AND date(o.created_at)<=date('now')) order_count,(SELECT coalesce(sum(total_cents),0) FROM orders o WHERE o.customer_id=lp.customer_id AND o.restaurant_id=lp.restaurant_id AND date(o.created_at)>=date('now','-90 days') AND date(o.created_at)<=date('now')) total_spend_cents,(SELECT count(*) FROM loyalty_transactions t WHERE t.loyalty_profile_id=lp.id AND t.type='redeem' AND date(t.created_at)>=date('now','-90 days') AND date(t.created_at)<=date('now')) reward_redemptions_count FROM loyalty_profiles lp WHERE lp.restaurant_id=? ORDER BY lp.points_balance DESC,lp.customer_id LIMIT 50",(rid,))
        return TaskSpec([call('get_top_loyalty_customers',restaurant_id=rid,days=90,limit=50)],[],[{'kind':'answer','value':'{}','name':'missing_stats'},{'kind':'answer','value':'{"customers":[]}','name':'empty_ranking'}],answer=resultref(),answer_expected=expected,answer_path=['customers'],answer_columns=['customer_id','points_balance','order_count','total_spend_cents','reward_redemptions_count'],notes='最近90天按模拟器UTC日界记录；不改动初始相对日期。')
    if i==9:
        candidates=query(db,"SELECT oi.id,oi.order_id FROM order_items oi JOIN orders o ON o.id=oi.order_id WHERE o.restaurant_id=? AND o.status='delivered' AND oi.name='Spicy Ramen' AND (lower(coalesce(o.meta,'')) LIKE '%spic%' OR lower(coalesce(o.internal_note,'')) LIKE '%issue%' OR EXISTS(SELECT 1 FROM order_internal_notes n WHERE n.order_id=o.id AND lower(n.note) LIKE '%spic%')) ORDER BY oi.id",(rid,))
        if not candidates:return TaskSpec([],[],status='needs_review',notes='无法从原始数据确认顾客投诉与Spicy Ramen订单项的关联，不能仅凭delivered状态猜测退款对象。')
        item=candidates[0];payment=scalar(db,'SELECT id FROM payments WHERE order_id=? ORDER BY id LIMIT 1',(item['order_id'],));note='Partial refund for incorrect spice level'
        body=dict(order_id=item['order_id'],order_item_id=item['id'],amount_cents=1200,reason='incorrect spice level',meta={'resolution':note})
        return TaskSpec([call('create_refund',payment_id=payment,body=body),call('add_order_internal_note',order_id=item['order_id'],body={'note':note})],[row_rule('refunds',{'payment_id':payment,'order_id':item['order_id'],'order_item_id':item['id'],'amount_cents':1200},new=True),Rule('refund metadata describes resolution',"SELECT count(*)=1 FROM refunds WHERE order_id=? AND order_item_id=? AND meta LIKE ? AND id NOT IN (SELECT id FROM initial.refunds)",[item['order_id'],item['id'],'%'+note+'%']),row_rule('order_internal_notes',{'order_id':item['order_id'],'note':note},new=True)],[mutate(0,'body.amount_cents',1500),{'kind':'omit','index':1,'name':'missing_resolution_note'}])
    raise ValueError(i)
