#!/usr/bin/env python3
"""Repair control transport/representation only; reuse valid executed controls."""
from pathlib import Path
from awm_new_pilot import DEFAULT_RUN,Pilot,assertion_query,freeze,task_spec
from harness_awm.io import db_fingerprint,read_json


async def repair(run,pilot=None):
    p=pilot or Pilot(run)
    repaired=[]
    for plan_path in sorted(p.run.glob('worlds/*/controls/plan.json')):
        dest=plan_path.parent
        old_result=dest/'result.json'
        if not old_result.exists() or read_json(old_result).get('passed'):
            continue
        folder=p.data/'worlds'/dest.parent.name
        plan=read_json(plan_path)
        task=task_spec(p.m,folder,0,read_json(folder/'tasks.json')['tasks'][0])
        inp=read_json(folder/'rubrics/00/original.json')
        rubric=read_json(folder/'rubrics/00/rubric.json')
        results={}
        for mode in ['complete','partial','unexecuted']:
            case=dest/mode
            r=read_json(case/'result.json')
            execution=r['execution']
            if mode=='unexecuted' and execution.get('error')=='RuntimeError: script_exhausted' and execution['tool_calls']==0:
                case=dest/'unexecuted.protocol_v2'
                action=dict(type='finish',status='incomplete',answer='I have not executed any requested changes.',reason='Deliberate negative control.')
                r=await p.episode(task,inp,rubric,case,[action,dict(action)])
            checks=[]
            for a in plan['assertions']:
                before,before_ok=assertion_query(case/'episode/initial.db',a['sql'],a['initial'])
                after,after_ok=assertion_query(case/'episode/final.db',a['sql'],a['initial' if mode=='unexecuted' else mode])
                checks.append(dict(obligation=a['obligation'],initial=before,final=after,passed=before_ok and after_ok))
            changed=db_fingerprint(case/'episode/initial.db')!=db_fingerprint(case/'episode/final.db')
            e=r.get('evaluation',{})
            runtime_ok=all(c['passed'] for c in checks) and (changed if mode!='unexecuted' else not changed)
            results[mode]=dict(checks=checks,episode_ref=str(case/'episode'),runtime_passed=runtime_ok,
                evaluation_status=e.get('evaluation_status'),business_label=e.get('business_label'),
                passed=runtime_ok and e.get('evaluation_status')=='valid' and e.get('business_label')==('pass' if mode=='complete' else 'fail'))
        saved=dest/'result.before_protocol_fix.json'
        if saved.exists():
            raise ValueError('control_fix_already_applied')
        old_result.rename(saved)
        freeze(old_result,dict(passed=all(v['passed'] for v in results.values()),cases=results,
            correction='Compare named or positional SQL rows by verified columns; repeat only the unexecuted control with the required V2 finish confirmation. No Agent episode or valid control replayed.'))
        repaired.append(dest.parent.name)
    return repaired
