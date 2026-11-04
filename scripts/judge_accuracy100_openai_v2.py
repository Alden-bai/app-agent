#!/usr/bin/env python3
"""Repair reference SQL ambiguity; preserve all actor attempts and first Judge results."""
import asyncio
import collections
import fcntl
import json
import shutil
import sqlite3
import time
from pathlib import Path

import judge_accuracy100 as base
import judge_accuracy100_openai as previous
from harness_awm.io import canonical, digest, file_hash, now, read_json, write_json


def strict_sql_evidence(initial, final, query):
    result = {'id': query['id'], 'sql': query['sql'], 'valid': False,
              'policy': 'All database tables must be explicitly qualified with initial. or final.'}
    try:
        with sqlite3.connect(':memory:', uri=True) as db:
            for alias, path in [('initial', initial), ('final', final)]:
                db.execute(f'ATTACH DATABASE ? AS {alias}', (Path(path).resolve().as_uri() + '?mode=ro',))
            # Empty shadow tables make SQLite resolve every unqualified name to
            # main, where the authorizer rejects it. CTEs and qualified joins work.
            names = set()
            for alias in ('final', 'initial'):
                tables = db.execute(f"SELECT name FROM {alias}.sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'").fetchall()
                for (name,) in tables:
                    if name in names: continue
                    names.add(name); quoted = '"' + name.replace('"','""') + '"'
                    db.execute(f'CREATE TABLE main.{quoted} AS SELECT * FROM {alias}.{quoted} LIMIT 0')
            db.execute('PRAGMA query_only=ON')
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
            unqualified = []
            def authorize(action, arg1, arg2, database, source):
                if action == sqlite3.SQLITE_READ and database not in {'initial','final'}:
                    unqualified.append(arg1); return sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_FUNCTION and (arg2 or '').lower() in {'load_extension','readfile','writefile'}:
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
            db.set_authorizer(authorize)
            deadline = time.monotonic()+3
            db.set_progress_handler(lambda: int(time.monotonic()>deadline),1000)
            try: cursor = db.execute(query['sql'])
            except sqlite3.Error:
                if unqualified: raise ValueError('Unqualified table reference; explicitly use initial.TABLE or final.TABLE, including every JOIN and subquery: '+','.join(unqualified))
                raise
            rows = cursor.fetchmany(101)
            result.update(valid=True, columns=[c[0] for c in cursor.description],
                rows=[[{'blob_hex':v.hex()} if isinstance(v,bytes) else v for v in row] for row in rows[:100]],
                truncated=len(rows)>100)
            while len(canonical(result).encode())>24000 and result['rows']:
                result['rows'].pop();result['truncated']=True
    except Exception as e: result['error']=str(e)
    return result


class CorrectedStudy(previous.OpenAIStudy):
    def __init__(self, run):
        super().__init__(run)
        base.sql_evidence = strict_sql_evidence
        amendment = self.run / 'reference_sql_amendment.json'
        if not amendment.exists():
            write_json(amendment, {'at':now(),'implementation':str(Path(__file__).resolve()),'sha256':file_hash(__file__),
                'reason':'Unqualified SQL resolved to initial database instead of the intended final state.',
                'fix':'Unqualified table reads rejected; initial/final must be explicit.',
                'actor_policy':'Reuse original episodes, no actor rerun.',
                'judge_policy':'Reuse original first Judge results; do not resample.',
                'reference_policy':'Affected labels re-established in fresh contexts with official outputs excluded. Some repairs occur after the first Judge; report this separately.'})
        else: assert read_json(amendment)['sha256']==file_hash(__file__)

    def affected(self, out):
        if not (out/'queries.json').exists():return False
        for q in read_json(out/'queries.json'):
            if not q.get('valid'):continue
            checked = strict_sql_evidence(out/'episode/initial.db',out/'episode/final.db',q)
            if 'Unqualified table reference' in checked.get('error',''):return True
        return False

    async def task(self, entry):
        result = await super().task(entry)
        out = self.run/'tasks'/entry['id']
        if not self.affected(out): return result
        async with self.env_sem:
            history = out/'assessment_history/v1';history.mkdir(parents=True,exist_ok=True)
            for name in ['result.json','reference_frozen.json']:
                target=history/name
                if not target.exists() and (out/name).exists():shutil.copyfile(out/name,target)
            repair = out/'reference_sql_v2'
            frozen = repair/'reference_frozen.json'
            if not frozen.exists():
                try:
                    task=base.TaskSpec.model_validate(read_json(out/'task_spec.json'))
                    reference=await self.reference(entry,task,out/'episode',repair)
                except Exception as exc:
                    reference={'label':'unknown','error':f'{type(exc).__name__}: {exc}'}
                reference.update(frozen_at=now(),blind_to_official=True,posthoc_reference_repair=True,
                    prior_reference_sha256=file_hash(history/'reference_frozen.json'))
                write_json(frozen,reference)
            reference=read_json(frozen)
            result.update(reference_label=reference['label'],reference_hash=file_hash(frozen),
                active_reference=str(frozen),reference_repaired_after_judge=(out/'official_evaluation.json').exists())
            for key in ['agreement','dispute','dispute_error']:result.pop(key,None)
            if reference['label'] in {'pass','fail'} and result['judge_valid']:
                result['agreement']=(reference['label']=='pass')==result['official_pass']
            write_json(out/'result.json',result)
            print(canonical({'reference_repaired':entry['task_id'],'label':reference['label'],'official_pass':result['official_pass']}),flush=True)
            self.report()
            return result

    def report(self):
        report=super().report()
        records=[read_json(p) for p in (self.run/'tasks').glob('*/result.json')]
        report['posthoc_reference_repair_count']=sum(r.get('reference_repaired_after_judge',False) for r in records)
        report['pending_reference_sql_repairs']=[r['task_id'] for r in records if self.affected(self.run/'tasks'/r['id']) and not r.get('active_reference')]
        report['limitations'].append('SQL ambiguity repaired; original Judge results reused. Posthoc repaired references are blind to Judge inputs/results but were frozen after original judgments.')
        write_json(self.run/'summary.json',report)
        return report

    async def run_all(self):
        # Interrupted official attempts stay unjudged; completed actors are reused.
        await asyncio.gather(*(self.task(t) for t in self.m['tasks'][:17]))
        report=self.report()
        print(canonical({'repair_checkpoint':True,'terminal':report['terminal_tasks'],'comparable':report['comparable']}),flush=True)
        await asyncio.gather(*(self.task(t) for t in self.m['tasks'][17:]))
        report=self.report()
        write_json(self.run/'completion.json',{'at':now(),'terminal_tasks':report['terminal_tasks'],'summary_sha256':file_hash(self.run/'summary.json'),'reference_sql_revision':2})
        return report


if __name__=='__main__':
    study=CorrectedStudy(previous.RUN)
    with (study.run/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(canonical(asyncio.run(study.run_all())),flush=True)
