"""One bounded, source-preserving reassessment of the remaining 108 records."""
import argparse
import asyncio
from collections import Counter, deque
from contextlib import ExitStack
import csv
import copy
import json
import errno
import fcntl
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sft_rate_limited_recheck_20261110 as previous
from scripts.sft_rate_recheck_inputs import merge_budget
from harness_awm.io import canonical
from harness_awm.judge_evidence import EvidenceStore, model_view

r = previous.r
PRIOR = previous.RUN
RUN = ROOT / 'runs/sft_remaining_recheck_20261110_v1'
DATA = Path('/root/autodl-tmp/awm-remaining-recheck-20261110-v1')
DRAIN = False
ADDITIONAL_PROMPT = """
输出前自检：queries每批最多12条；finish.task_id必须逐字使用rubric.task_id，items覆盖全部条目且各一次。
每条unknown字段必须显式存在；任一证明established=true时unknown=null，两者false时unknown为完整对象。不可通过机械翻转布尔值消除校验错误。
unresolved_checks的id及affected_items集合必须原样保留rubric.unresolved中的定义，即使只剩部分事实未解决也不可缩小集合。
rubric_issues.source_ids只能取rubric.source_ids中的字符串，不能填E开头的执行证据ID或自行构造来源名。
分页用原始source_evidence_id和返回的next_offset续读，不分页读取已有分页，不重复读取相同来源的相同offset。先利用已取得的证据，再为具体缺失事实查询。
只有核对原题及契约后能指出固定规则的实际增删或矛盾才报rubric_issues。工具不能直接观测某个原题要求本身并不证明规则错误；应区分真实歧义、不可获得事实、仍可补查的证据缺口。
维持原评分义务，使用实际执行时钟；不得为了得到分数伪造事实、删除未知项或放宽规则。
"""


def sealed_copy(source, target):
    source, target = Path(source).resolve(), Path(target)
    seal = r.verify_seal(source)
    temporary = target.with_name(target.name + '.preparing')
    temporary.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in set(seal) | {'evidence_seal.json', 'episode_result.json'}:
        src, dst = (source/name).resolve(), temporary/name
        if not src.is_relative_to(source) or not dst.resolve().is_relative_to(temporary.resolve()):
            raise ValueError('evidence_path_escape')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            if name in seal and name not in ('episode_result.json', 'evidence_seal.json'):
                try:
                    os.link(src, dst)
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    shutil.copy2(src, dst)
                    copied.append(name)
            else:
                shutil.copy2(src, dst)
    if r.verify_seal(temporary) != seal:
        raise ValueError('copied_seal_mismatch')
    os.rename(temporary, target)
    return dict(original_episode=str(source), original_result_sha256=r.file_hash(source/'episode_result.json'),
                original_seal_sha256=r.file_hash(source/'evidence_seal.json'), cross_device_copies=copied)


class ExpandedEvidenceStore(EvidenceStore):
    def read_evidence(self, evidence_id, offset=0):
        # Preserve the original validation and character-offset semantics.
        super().read_evidence(evidence_id, offset)
        record = self.records[evidence_id]
        text = canonical(record['content'])
        allowance = min(8000, self.max_bytes - 256)
        chunk = text[offset:].encode()[:allowance].decode('utf-8', errors='ignore')
        end = offset + len(chunk)
        return dict(source_evidence_id=evidence_id, source_kind=record['kind'], offset=offset,
                    next_offset=end if end < len(text) else None, total_chars=len(text), text=chunk), (
                    offset > 0 or end < len(text) or record.get('truncated', False))


def expanded_view(records, max_bytes=80000):
    # The runtime's old 40k argument is deliberately superseded by this policy.
    limit = 80000
    views = model_view(records, max_bytes=0)
    latest = {}
    for eid in reversed(records):
        kind = records[eid]['kind']
        if kind in ('runtime', 'trajectory', 'final_answer'):
            latest.setdefault(kind, eid)
    priority = list(latest.values())
    used = 0
    for eid in priority + [x for x in reversed(records) if x not in priority]:
        text = canonical(records[eid]['content'])
        cap = 3200 if eid in priority else 10000
        preview = text.encode()[:max(0, min(cap, limit-used))].decode('utf-8', errors='ignore')
        complete = preview == text
        views[eid].update(content=records[eid]['content'] if complete else dict(
            preview=preview, total_chars=len(text), read_with='read_evidence'), view_truncated=not complete)
        used += len(canonical(views[eid]).encode())
    return views


class FullOutputTransport(previous.Transport):
    async def request(self, config, kwargs, budget, key, stage, folder, **kw):
        # backend.fits_context counts UTF-8 bytes as tokens. Use the same
        # tokenizer + 10% margin as the provider rate reservation instead.
        wire = copy.deepcopy(kwargs)
        ceiling = min(config.context_limit or 95000, 95000)
        upper = previous.base.quota_tokens(wire['messages'], 8192)
        for cap in (5000, 2000, 800, 0):
            if upper <= ceiling:
                break
            for message in wire['messages']:
                if message['role'] != 'user':
                    continue
                try:
                    value = json.loads(message['content'])
                except (ValueError, TypeError):
                    continue
                if not isinstance(value, dict) or not isinstance(value.get('evidence'), dict):
                    continue
                for view in value['evidence'].values():
                    content = view['content']
                    text = content['preview'] if view.get('view_truncated') else canonical(content)
                    total = content['total_chars'] if view.get('view_truncated') else len(text)
                    preview = text.encode()[:cap].decode('utf-8', errors='ignore')
                    if len(preview) < len(text):
                        view.update(content=dict(preview=preview,total_chars=total,read_with='read_evidence'),view_truncated=True)
                message['content'] = canonical(value)
            upper = previous.base.quota_tokens(wire['messages'], 8192)
        if upper > ceiling:
            raise ValueError('tokenized_request_exceeds_safe_context')
        r.write_json(Path(folder)/'context_check.json', dict(
            tokenizer_reservation=upper, configured_context_limit=config.context_limit,
            byte_estimate=previous.base.estimate_tokens(wire['messages']), output_tokens=8192,
            evidence_view_reduced=wire['messages'] != kwargs['messages']))
        api_config = config.model_copy(update={'context_limit': None})
        return await self.exact(api_config, dict(wire, max_completion_tokens=8192),
                                budget, key, stage, folder, **kw)


def hashes():
    return {**previous.source_hashes(), str(Path(__file__).relative_to(ROOT)): r.file_hash(__file__)}


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    jobs = RUN/'judge_jobs'
    if not jobs.exists():
        jobs.symlink_to(DATA, target_is_directory=True)
    assert jobs.resolve() == DATA.resolve()
    if (RUN/'plan.json').exists():
        plan = r.read_json(RUN/'plan.json')
        assert r.file_hash(RUN/'plan.json') == r.read_json(RUN/'plan.sha256.json')['sha256']
        assert plan['source_hashes'] == hashes(), 'implementation_changed'
        return plan
    inventory = PRIOR/'remaining_unscored_inventory.csv'
    with inventory.open(encoding='utf-8-sig') as f:
        cohort = list(csv.DictReader(f))
    assert len(cohort) == 108
    historical = {x['source_episode']: x for x in r.recovery.collect_repairs()}
    seen = set()
    for row in cohort:
        source = Path(row['source_episode'])
        key = r.digest([row['candidate'], row['task_instance_id'], str(source)])[:24]
        assert key not in seen
        seen.add(key)
        latest = Path(row['latest_episode'])
        assert r.evaluation_score(r.read_json(latest/'episode_result.json')) is None
        task = r.TaskSpec.model_validate(r.read_json(source/'task.json'))
        assert task.task_instance_id == row['task_instance_id']
        paths = [source.parents[2]/'budget.sqlite']
        if str(source) in historical:
            paths.append(r.REPAIRS/'judge_jobs_v1'/historical[str(source)]['key']/'budget.sqlite')
        paths += [r.RUN/'judge_jobs'/key/'budget.sqlite', PRIOR/'judge_jobs'/key/'budget.sqlite']
        row.update(key=key, budget_sources=list(dict.fromkeys(str(x) for x in paths if x.exists())),
                   source_result_sha256=r.file_hash(source/'episode_result.json'),
                   source_seal_sha256=r.file_hash(source/'evidence_seal.json'))
    priority = {'not_assessed': 0, 'APIConnectionError': 1, 'judge_output_invalid_after_repair': 2,
                'required_evidence_incomplete': 3, 'query_round_budget_exhausted': 4, 'rubric_error': 5}
    cohort.sort(key=lambda x: (priority[x['reason']], x['candidate'], x['key']))
    config = r.read_json(PRIOR/'plan.json')['config']
    config['judge_v2']['query_rounds'] = 16
    cfg = r.RunConfig.model_validate(config)
    # Validate all inputs and merge budgets before any paid request.
    for row in cohort:
        job = jobs/row['key'];job.mkdir(exist_ok=True)
        source = Path(row['source_episode'])
        task = r.TaskSpec.model_validate(r.read_json(source/'task.json'))
        r.input_from_manifest(source, task, cfg.judge_inputs_manifest)
        bp = job/'budget.sqlite'
        meta = r.read_json(bp.with_suffix('.baseline.json')) if bp.exists() else merge_budget(row['budget_sources'], bp)
        row['baseline_requests'] = meta['baseline']['physical_requests']
        row['max_cumulative_requests'] = row['baseline_requests'] + 32
        meta['max_cumulative_requests'] = row['max_cumulative_requests']
        meta['new_round_request_allowance'] = 32
        r.write_json(job/'baseline_budget.json', meta)
    plan = dict(at=r.now(), cohort=cohort, config=config, source_hashes=hashes(),
                input_inventory_sha256=r.file_hash(inventory), parent_plan_sha256=r.file_hash(PRIOR/'plan.json'),
                policy=dict(tpm=2000000, rpm=600, workers=16, new_physical_requests_per_episode=32,
                    semantic_attempts=1, query_rounds=16, review_rounds=4, format_repairs=2,
                    page_bytes=8000, evidence_view_bytes=80000, output_tokens=8192,
                    prompt_addition=ADDITIONAL_PROMPT, original_rubric_unchanged=True))
    r.write_json(RUN/'plan.json', plan)
    r.write_json(RUN/'plan.sha256.json', {'sha256': r.file_hash(RUN/'plan.json')})
    r.write_json(RUN/'transport_settings.json', dict(rpm=600, workers=16, target_tpm=2000000))
    return plan


async def assess(row, config, runtime):
    job = RUN/'judge_jobs'/row['key'];source=Path(row['source_episode']);target=job/'episode'
    assert r.file_hash(source/'episode_result.json') == row['source_result_sha256']
    assert r.file_hash(source/'evidence_seal.json') == row['source_seal_sha256']
    if not target.exists():
        r.write_json(job/'provenance.json', sealed_copy(source, target))
    baseline=r.read_json(job/'baseline_budget.json')['baseline']
    budget=r.Budget(job/'budget.sqlite', row['max_cumulative_requests'], None)
    task=r.TaskSpec.model_validate(r.read_json(target/'task.json'))
    bundle=r.input_from_manifest(target, task, config.judge_inputs_manifest)
    assessment=await runtime.evaluate_input(bundle, config, budget, asyncio.Semaphore(1), target/'evaluations_v2')
    if any('TransportPaused' in str(e) for e in assessment.get('errors', [])):
        r.write_json(job/'deferred.json', assessment);return None
    outcome=r.settle_reward(assessment, bundle, task, config)
    r.write_json(target/'rewards_v2'/(r.digest(outcome)+'.json'), outcome)
    result=r.apply_evaluation(target, outcome);r.verify_seal(target)
    assert r.file_hash(source/'episode_result.json') == row['source_result_sha256']
    cumulative=budget.summary()
    record=dict(candidate=row['candidate'],task_id=row['task_id'],task_instance_id=task.task_instance_id,
        source_episode=str(source),episode=str(target),previous_failure=row['reason'],score=r.evaluation_score(result),
        errors=assessment['errors'],evaluation_status=result.get('evaluation_status'),
        rubric_issues=assessment.get('rubric_issues', []),assessment=assessment['attempt'],at=r.now(),
        usage={k:v-baseline.get(k,0) for k,v in cumulative.items()},cumulative_usage=cumulative)
    r.write_json(job/'recovery.json',record)
    return record


async def drive(plan):
    previous.RUN=RUN
    config=r.RunConfig.model_validate(plan['config'])
    if not os.environ.get(config.judge_api.api_key_env,'').strip():
        raise ValueError('missing_api_key_environment_variable')
    fresh=not (RUN/'provider_rate.sqlite').exists()
    gate=previous.SmoothGate(RUN/'provider_rate.sqlite')
    if fresh:
        with sqlite3.connect(gate.path) as db:db.execute('UPDATE pacing SET tpm=2000000 WHERE id=1')
    with sqlite3.connect(gate.path) as db:gate.tpm=db.execute('SELECT tpm FROM pacing WHERE id=1').fetchone()[0]
    runtime=r.recovery_judge()
    transport=FullOutputTransport(runtime.api_request,gate,RUN)
    runtime.api_request=transport.request
    runtime.PROMPT += ADDITIONAL_PROMPT
    runtime.EvidenceStore=ExpandedEvidenceStore
    runtime.model_view=expanded_view
    original_hash=runtime.config_hash
    runtime.config_hash=lambda cfg:r.digest({'original':original_hash(cfg),'round_policy':r.file_hash(RUN/'plan.json')})
    records={}
    for row in plan['cohort']:
        path=RUN/'judge_jobs'/row['key']/'recovery.json'
        if path.exists():records[row['key']]=r.read_json(path)
    queue=deque(x for x in plan['cohort'] if x['key'] not in records)
    active={};fatal=[]
    def report(final=False):
        valid=sum(x['score'] is not None for x in records.values())
        stage='needs_attention' if fatal or transport.stopped else ('complete' if len(records)==108 else 'paused' if final else 'draining' if DRAIN else 'rescoring')
        r.write_json(RUN/'status.json',dict(at=r.now(),pid=os.getpid(),stage=stage,total=108,completed=len(records),
            valid=valid,needs_review=len(records)-valid,remaining=108-len(records),active=active,workers=16,
            rate=gate.snapshot(),fatal=fatal,transport_stop=transport.stopped,
            total_validation_scored=2837+valid,scores=dict(Counter(str(x['score']) for x in records.values() if x['score'] is not None))))
        r.write_json(RUN/'results.json',list(records.values()))
        r.write_json(RUN/'unresolved.json',[x for x in records.values() if x['score'] is None])
    async def reporter():
        while True:report();await asyncio.sleep(5)
    async def worker(index):
        while queue and not (DRAIN or fatal or transport.stopped):
            if min(shutil.disk_usage(RUN).free,shutil.disk_usage(DATA).free)<1024**3:
                fatal.append('disk_below_1GiB');return
            row=queue.popleft();active[index]={'task_id':row['task_id'],'reason':row['reason']}
            try:
                record=await assess(row,config,runtime)
                if record is not None:records[row['key']]=record
            except Exception as error:fatal.append(type(error).__name__+':'+str(error))
            finally:active.pop(index,None)
    reporter_task=asyncio.create_task(reporter())
    try:await asyncio.gather(*(worker(i) for i in range(16)))
    finally:
        reporter_task.cancel()
        try:await reporter_task
        except asyncio.CancelledError:pass
        report(final=True)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False)
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    def drain(*_):
        global DRAIN
        DRAIN=True
    signal.signal(signal.SIGUSR1,drain);signal.signal(signal.SIGTERM,drain)
    RUN.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for path in (RUN/'run.lock',PRIOR/'run.lock',r.RUN/'judge.lock',r.REPAIRS/'supervisor.lock'):
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare()
        if args.mode=='prepare':print({'prepared':len(plan['cohort']),'by_reason':dict(Counter(x['reason'] for x in plan['cohort']))});return
        try:asyncio.run(drive(plan))
        except Exception as error:
            r.write_json(RUN/'status.json',dict(at=r.now(),stage='needs_attention',reason=type(error).__name__+':'+str(error)));raise


if __name__=='__main__':main()
