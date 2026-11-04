"""Archive batch04's verified zero-session CLI initialization failures offline.

This narrow repair does not infer completion from a timeout and cannot accept an
attempt with any thread, event, response, or different startup error.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import rubric_luna_batch04 as b


def request_key(identity):
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def files(folder):
    return {p.name:b.digest(p) for p in folder.iterdir() if p.is_file()}


def main():
    b.configure()
    with b.legacy.locked(b.RUN/'.runner.lock',blocking=False):
        journal = b.RUN/'startup_reconciliation'
        plan_path = journal/'plan.json'
        committed = journal/'committed.json'
        if committed.exists():
            print(json.dumps(b.read(committed),ensure_ascii=False),flush=True)
            return
        state_path = b.RUN/'scheduler_state.json'
        state = b.read(state_path)
        if plan_path.exists():
            plan = b.read(plan_path)
        else:
            paths = sorted(b.RUN.glob('tasks/*/round_*/*/identity.json'))
            if len(paths) != 3 or state.get('stop_reason') != 'model_configuration':
                raise RuntimeError('unexpected_startup_attempt_set')
            if state.get('active') or len(state.get('remote_holds',{})) != 3:
                raise RuntimeError('unexpected_execution_holds')
            rows = []
            for path in paths:
                folder = path.parent
                identity = b.read(path)
                end = b.read(folder/'completed.json')
                usage = b.read(folder/'usage.json')
                stderr = (folder/'stderr.txt').read_text()
                if not (end.get('identity') == identity and end.get('status') == 'model_unavailable'
                    and end.get('returncode') == 1 and end.get('thread_id') is None
                    and end.get('server_completion_evidence') is False
                    and end.get('response_sha256') is None
                    and usage.get('usage_unknown') is True and usage.get('error') == 'no_thread'
                    and 'failed to initialize in-process app-server client: Read-only file system' in stderr
                    and (folder/'events.jsonl').stat().st_size == 0
                    and not (folder/'response.txt').exists()
                    and not (folder/'response.pending.txt').exists()):
                    raise RuntimeError('not_proven_zero_model_session:'+str(folder))
                key = request_key(identity)
                if key not in state['remote_holds']:
                    raise RuntimeError('hold_identity_mismatch:'+key)
                rows.append(dict(folder=str(folder.relative_to(b.RUN)),key=key,identity=identity,files=files(folder)))
            if {r['key'] for r in rows} != set(state['remote_holds']):
                raise RuntimeError('extra_remote_hold')
            plan = dict(run_id=b.RUN_ID,manifest_sha256=b.digest(b.RUN/'manifest.json'),
                        evidence='CLI initialization failed before thread creation; empty events and no response',
                        rows=rows,state_before=state,verified_model_sessions=0)
            b.freeze(plan_path,plan)
        if plan['run_id'] != b.RUN_ID or plan['manifest_sha256'] != b.digest(b.RUN/'manifest.json'):
            raise RuntimeError('recovery_plan_wrong_batch')
        for row in plan['rows']:
            source = b.RUN/row['folder']
            target = b.RUN/'interrupted_attempts'/row['key']
            current = source if source.exists() else target
            if not current.exists() or files(current) != row['files']:
                raise RuntimeError('archive_source_changed:'+str(current))
            if source.exists():
                if target.exists():
                    raise RuntimeError('archive_destination_exists:'+str(target))
                b.durable_mkdir(target.parent)
                os.rename(source,target)
                b.fsync_dir(source.parent)
                b.fsync_dir(target.parent)
        if state.get('remote_holds'):
            if state != plan['state_before']:
                raise RuntimeError('scheduler_state_changed_during_recovery')
            state['remote_holds'] = {}
            state['active'] = {}
            state['stop_reason'] = None
            state['consecutive_infrastructure_failures'] = 0
            state['recent_infrastructure_failures'] = []
            state['calls_launched'] = 0
            scale = state.get('autoscale') or {}
            scale.update(window_start=time.time(),rows=[],windows=[],stable={},bad_windows=0,
                         level_since=time.time(),level_calls=0,level_failed=False,
                         last_decision='verified_zero_session_startup_reconciled')
            state['autoscale'] = scale
            b.snapshot(state_path,state)
            b.append(b.RUN/'reconciliation_events.jsonl',dict(at=b.now(),
                reason='verified_zero_session_cli_startup_failure',archived=[r['key'] for r in plan['rows']],
                prior_state_sha256=b.digest(plan_path),paid_model_calls=0))
        elif state.get('stop_reason') is not None:
            raise RuntimeError('unexpected_recovery_state')
        m,_ = b.legacy.execution_manifest(b.RUN)
        lookup = {t['task_id']:t for t in m['tasks']}
        with b.legacy.locked(b.REGISTRY/'.lock'):
            batches,_ = b.legacy.replay()
            record = b.legacy.batch_record(m,batches[b.RUN_ID])
            record['status'] = 'reserved'
            rows = [b.legacy.task_record(m,lookup[r['identity']['task_id']],b.RUN,b.RUN_ID)
                    for r in plan['rows']]
            b.legacy.commit('已核实三个CLI启动失败未建立模型会话；原请求归档后恢复占位',[record],rows)
        result = dict(at=b.now(),run_id=b.RUN_ID,archived=len(plan['rows']),
                      verified_model_sessions=0,paid_model_calls=0,plan_sha256=b.digest(plan_path))
        b.freeze(committed,result)
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
