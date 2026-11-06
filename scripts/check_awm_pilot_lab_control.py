"""Source-derived controls for accepted lab task 03; preserve raw Judge outputs."""
import asyncio,fcntl,os
from pathlib import Path
from awm_new_pilot import *
async def main():
 p=Pilot(DEFAULT_RUN); f=p.data/'worlds/laboratory_reagent_qc_release_1'; dest=p.run/'worlds'/f.name/'controls.task03'
 if (dest/'result.json').exists():return
 task=task_spec(p.m,f,3,read_json(f/'tasks.json')['tasks'][3]);inp=read_json(f/'rubrics/03/original.json');rubric=read_json(f/'rubrics/03/rubric.json')
 cases={}
 read=dict(name='get_lot_storage_history',arguments={'lot_id':12})
 move=dict(name='create_lot_storage_movement',arguments=dict(lot_id=12,location_id=2,recorded_min_temp_c=-22,recorded_max_temp_c=-18,started_at='2026-11-04T12:00:00Z'))
 freeze(dest/'plan.json',dict(task=task.instruction,source_task_sha256=file_hash(f/'tasks.json'),initial_sha256=file_hash(f/'initial.db'),complete_calls=[read,move,read],partial_calls=[read],interpretation='The -20 C storage class uses the public compatible-range definition: reagent [-25,-15], available freezer [-22,-18]. The rubric instead demands both endpoints exactly -20; this control tests that source/asset mismatch without changing it.'))
 for mode,calls in [('complete',[read,move,read]),('partial',[read]),('unexecuted',[])]:
  actions=[]
  for c in calls:actions.extend([dict(type='get_tool_schema',name=c['name']),dict(type='call_tool',**c)])
  action=dict(type='finish',status='completed' if mode=='complete' else 'incomplete',answer='Moved TAQ-801 to available -20 C Freezer FZ-20-01 without changing release status.' if mode=='complete' else 'Requested move has not been performed.',**({} if mode=='complete' else dict(reason='Deliberate negative control.')))
  actions.append(action)
  if not calls:actions.append(dict(action))
  r=await p.episode(task,inp,rubric,dest/mode,actions)
  sql='SELECT l.release_status,h.location_id,h.recorded_min_temp_c,h.recorded_max_temp_c FROM reagent_lots l JOIN lot_storage_history h ON h.lot_id=l.id WHERE l.id=12 AND h.ended_at IS NULL'
  initial=query(dest/mode/'episode/initial.db',sql);final=query(dest/mode/'episode/final.db',sql)
  expected=[[initial[0][0],2,-22.0,-18.0]] if mode=='complete' else initial
  e=r.get('evaluation',{})
  cases[mode]=dict(initial=initial,final=final,sql=sql,expected=expected,db_assertions_passed=final==expected,expected_business_label='pass' if mode=='complete' else 'fail',evaluation_status=e.get('evaluation_status'),business_label=e.get('business_label'),episode_ref=str(dest/mode/'episode'))
 freeze(dest/'result.json',dict(cases=cases,passed=all(c['db_assertions_passed'] and c['evaluation_status']=='valid' and c['business_label']==c['expected_business_label'] for c in cases.values())))
 print(canonical(cases),flush=True)
if __name__=='__main__':
 from dotenv import dotenv_values
 if not os.environ.get('OPENAI_API_KEY'):os.environ['OPENAI_API_KEY']=dotenv_values(ROOT/'.env',interpolate=False)['OPENAI_API_KEY']
 with (DEFAULT_RUN/'runner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  asyncio.run(main())
