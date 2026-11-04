from datetime import date,timedelta
from awm_audit.common import query
from awm_audit.oracle import Rule,TaskSpec,call,mutate,row_rule,scalar,sqlref,one


def latest(table,where='1'):
    return sqlref(f'SELECT max(id) FROM {table} WHERE {where}')


def build(i,db):
    uid=1
    city=scalar(db,'SELECT city FROM users WHERE id=1')
    photo=scalar(db,"SELECT id FROM media WHERE owner_id=1 AND media_type='photo' ORDER BY created_at DESC,id DESC LIMIT 1")
    if i == 0:
        body=dict(content='Excited to start 2025 by learning web automation and building my first side project! #NewYear #Coding',audience='Friends')
        return TaskSpec([call('create_post',payload=body)],[row_rule('posts',{'author_id':1,**body},new=True)],[mutate(0,'payload.audience','Public'),mutate(0,'payload.content','Excited to start 2025')])
    if i == 1:
        friend=scalar(db,"SELECT id FROM users WHERE full_name='Alex Johnson'")
        media=one(db,'SELECT * FROM media WHERE id=?',(photo,))
        body=dict(media_type='photo',url=media['url'],mime_type=media['mime_type'],caption='Hiking at Yosemite last weekend—nature reset!',location_name='Yosemite National Park')
        calls=[call('create_media',payload=body),call('create_photo_tag',payload={'media_id':latest('media'),'tagged_user_id':friend})]
        rules=[row_rule('media',{'owner_id':1,**body},new=True),Rule('tag on newly uploaded photo',"SELECT count(*)=1 FROM photo_tags t JOIN media m ON m.id=t.media_id WHERE m.id NOT IN (SELECT id FROM initial.media) AND m.caption=? AND t.tagged_user_id=?",[body['caption'],friend])]
        return TaskSpec(calls,rules,[mutate(0,'payload.location_name','Riverside Park'),{'kind':'omit','index':1,'name':'missing_friend_tag'}],notes='用户未指定照片文件；模拟上传复用初态最新照片URL，核对新media记录、地点、说明和好友标签。')
    if i == 2:
        friends=[x['id'] for x in query(db,"SELECT u.id FROM users u WHERE u.city=? AND u.id IN (SELECT friend_id FROM friendships WHERE user_id=1 AND status='friends' UNION SELECT user_id FROM friendships WHERE friend_id=1 AND status='friends') ORDER BY u.id",(city,))]
        body=dict(title='Community Park Clean-Up Day',description='Join us to help clean up our neighborhood park; bags and gloves provided.',location_name='Riverside Park',start_time='2025-03-15T09:00:00',end_time='2025-03-15T13:00:00',audience='Public')
        calls=[call('create_event',payload=body)]+[call('create_event_invitation',payload={'event_id':latest('events'),'user_id':f,'status':'invited'}) for f in friends]
        values={k:v for k,v in body.items() if k not in ['start_time','end_time']};values['creator_id']=1
        rules=[row_rule('events',values,new=True),Rule('event times and exact city friends',"SELECT datetime(start_time)='2025-03-15 09:00:00' AND datetime(end_time)='2025-03-15 13:00:00' AND (SELECT count(DISTINCT user_id) FROM event_invitations WHERE event_id=e.id)=? AND NOT EXISTS(SELECT user_id FROM event_invitations WHERE event_id=e.id EXCEPT SELECT value FROM json_each(?)) FROM events e WHERE id=(SELECT max(id) FROM events)",[len(friends),__import__('json').dumps(friends)])]
        return TaskSpec(calls,rules,[mutate(0,'payload.audience','Private'),{'kind':'omit','index':1,'name':'missing_invitations'}])
    if i == 3:
        calls=[call('get_my_most_recent_media',media_type='photo'),call('create_story',payload={'media_id':photo,'audience':'Friends'}),call('create_story_overlay',payload={'story_id':latest('stories'),'text':'Weekend vibes','font_color':'#FFFFFF','position':'bottom'})]
        return TaskSpec(calls,[row_rule('stories',{'author_id':1,'media_id':photo,'audience':'Friends'},new=True),Rule('overlay belongs to new story',"SELECT count(*)=1 FROM story_overlays o JOIN stories s ON s.id=o.story_id WHERE s.id NOT IN (SELECT id FROM initial.stories) AND o.text='Weekend vibes' AND lower(o.font_color) IN ('white','#ffffff') AND o.position='bottom'")],[mutate(1,'payload.audience','Public'),mutate(2,'payload.position','top')])
    if i == 4:
        maria=scalar(db,"SELECT id FROM users WHERE full_name='Maria Lopez'")
        conv=scalar(db,"SELECT conversation_id FROM conversation_participants WHERE user_id=? AND conversation_id IN (SELECT conversation_id FROM conversation_participants WHERE user_id=1)",(maria,))
        msg=scalar(db,"SELECT id FROM messages WHERE conversation_id=? AND sender_id=? ORDER BY created_at DESC,id DESC LIMIT 1",(conv,maria))
        text='Hey, are we still on for dinner this Friday at 7 PM?'
        calls=[call('create_message',payload={'conversation_id':conv,'content':text}),call('create_message_reaction',payload={'message_id':msg,'reaction':'👍'})]
        return TaskSpec(calls,[row_rule('messages',{'sender_id':1,'conversation_id':conv,'content':text},new=True),Rule('reaction to her most recent message',"SELECT EXISTS(SELECT 1 FROM message_reactions WHERE user_id=1 AND message_id=? AND reaction IN ('👍','thumbs_up'))",[msg])],[mutate(0,'payload.content','Dinner cancelled.'),mutate(1,'payload.reaction','❤️')])
    if i == 5:
        body=dict(name='Sunrise Bakes',category='Bakery',description='Artisan breads, pastries, and coffee in downtown.',address='123 Main St, Springfield',city=city)
        return TaskSpec([call('create_page',payload=body),call('create_page_media',payload={'page_id':latest('pages'),'media_id':photo,'is_profile_picture':True})],[row_rule('pages',{'owner_id':1,**body},new=True),Rule('latest photo on new page',"SELECT count(*)=1 FROM page_media p WHERE page_id=(SELECT max(id) FROM pages) AND media_id=? AND is_profile_picture=1",[photo])],[mutate(0,'payload.category','Restaurant'),{'kind':'omit','index':1,'name':'missing_profile_picture'}])
    if i == 6:
        body=dict(title='Used IKEA MALM Desk - White',price=60,condition='Good',location_city=city,description='Minor scratches on the surface, no structural damage, pickup only.',category_id=scalar(db,"SELECT id FROM marketplace_categories WHERE name='Furniture'"))
        return TaskSpec([call('create_marketplace_listing',payload=body)],[row_rule('marketplace_listings',{'seller_id':1,**body},new=True)],[mutate(0,'payload.price',80),mutate(0,'payload.condition','New')])
    if i == 7:
        vids=[x['id'] for x in query(db,"SELECT id FROM media WHERE owner_id=1 AND media_type='video' AND height>width ORDER BY created_at DESC,id DESC LIMIT 2")]
        if len(vids)!=2:raise ValueError('Fewer than two portrait videos in original data')
        body=dict(caption='Training progress so far! #RunningJourney',audience='Public')
        calls=[call('list_my_recent_videos',limit=100),call('create_reel',payload=body)]+[call('create_reel_clip',payload={'reel_id':latest('reels'),'media_id':v,'clip_order':j+1,'start_time_seconds':0,'end_time_seconds':5}) for j,v in enumerate(vids)]+[call('create_reel_text_overlay',payload={'reel_id':latest('reels'),'text':'From couch to 5K in 4 weeks','start_time_seconds':0,'end_time_seconds':3,'position':'top'})]
        rules=[row_rule('reels',{'creator_id':1,**body},new=True),Rule('both portrait clips trimmed to five seconds',"SELECT count(*)=2 AND sum(end_time_seconds-start_time_seconds=5 AND media_id IN (?,?))=2 FROM reel_clips WHERE reel_id=(SELECT max(id) FROM reels)",vids),Rule('opening text',"SELECT count(*)=1 FROM reel_text_overlays WHERE reel_id=(SELECT max(id) FROM reels) AND start_time_seconds=0 AND text='From couch to 5K in 4 weeks'")]
        return TaskSpec(calls,rules,[mutate(1,'payload.audience','Friends'),mutate(2,'payload.end_time_seconds',10)])
    if i == 8:
        ids=[scalar(db,'SELECT id FROM users WHERE full_name=?',(n,)) for n in ['John Smith','Priya Patel']]
        body=dict(name='Remote Developers Study Group',description="A space for remote devs to share resources, co-work, and review each other's code.",privacy='Private')
        return TaskSpec([call('create_group',payload=body)]+[call('create_group_member',payload={'group_id':latest('groups'),'user_id':u,'role':'member'}) for u in ids],[row_rule('groups',{'owner_id':1,**body},new=True),Rule('both friends in new group',"SELECT count(DISTINCT user_id)=2 FROM group_members WHERE group_id=(SELECT max(id) FROM groups) AND user_id IN (?,?)",ids)],[mutate(0,'payload.privacy','Public'),{'kind':'omit','index':2,'name':'missing_second_member'}])
    if i == 9:
        p=one(db,"SELECT id,address FROM pages WHERE name='Sunrise Bakes' AND owner_id=1")
        photos=query(db,"SELECT id,caption,location_name FROM media WHERE owner_id=1 AND media_type='photo' AND location_name='Sunrise Bakes' AND (lower(caption) LIKE '%sourdough%' OR lower(caption) LIKE '%bread%' OR lower(caption) LIKE '%pastr%') ORDER BY created_at DESC,id DESC LIMIT 1")
        if not photos:
            return TaskSpec([],[],status='needs_review',notes='最新product photo缺乏明确标签，不能自动将任意最新照片认定为产品照片。')
        start=date.today()+timedelta(days=(7-date.today().weekday())%7 or 7);end=start+timedelta(days=7)
        body=dict(page_id=p['id'],name='Audit traffic campaign',daily_budget=10,start_date=start.isoformat(),end_date=end.isoformat())
        calls=[call('create_ad_campaign',payload=body),call('create_ad_target',payload={'campaign_id':latest('ad_campaigns'),'min_age':21,'max_age':55,'radius_miles':10,'location_address':p['address']}),call('create_ad_creative',payload={'campaign_id':latest('ad_campaigns'),'media_id':photos[0]['id'],'primary_text':'Fresh sourdough every morning—order online for pickup!','destination_url':'https://sunrisebakes.com/order'})]
        rules=[row_rule('ad_campaigns',{'page_id':p['id'],'objective':'traffic','daily_budget':10},new=True),Rule('next Monday and seven days',"SELECT date(start_date)=? AND julianday(end_date)-julianday(start_date)=7 FROM ad_campaigns WHERE id=(SELECT max(id) FROM ad_campaigns)",[start.isoformat()]),row_rule('ad_targets',{'min_age':21,'max_age':55,'radius_miles':10,'location_address':p['address']},where='campaign_id=(SELECT max(id) FROM ad_campaigns)'),row_rule('ad_creatives',{'media_id':photos[0]['id'],'primary_text':'Fresh sourdough every morning—order online for pickup!','destination_url':'https://sunrisebakes.com/order'},where='campaign_id=(SELECT max(id) FROM ad_campaigns)')]
        return TaskSpec(calls,rules,[mutate(0,'payload.daily_budget',20),mutate(1,'payload.radius_miles',20)])
    raise ValueError(i)
