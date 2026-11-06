#!/usr/bin/env python3
"""Budgeted new-world development pilot; never imports historical answers.

prepare is offline. run resumes immutable requests, but never replays a started
episode. The pilot gate must pass before the remaining eight worlds are built.
"""
import argparse
import ast
import asyncio
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agent-world-model"))

from awm import prompts
from awm.tools import format_db_schema, normalize_scenario_name
from harness_awm.backend import APIBackend, Budget, ScriptedBackend, api_request
from harness_awm.contracts import APIConfig, RunConfig, TaskSpec
from harness_awm.environment import AWMEnvAdapter, worker
from harness_awm.io import canonical, db_check, db_fingerprint, digest, file_hash, read_json, rows, snapshot, write_json
from harness_awm.judge_inputs import prepare_input
from harness_awm.judge_runtime import evaluate_input
from harness_awm.registry import load_registry
from harness_awm.runner import EpisodeRunner
from harness_awm.protocol_v2 import fits_context, estimate_tokens
import rubric_d1 as d1

DEFAULT_RUN = ROOT / "runs/awm_new_pilot100_20261104_v1"
DEFAULT_DATA = ROOT / "data/awm-new-pilot100-20261104-v1"
BOUNDARIES = ROOT / "configs/rubric/d1_repair100_v1/business_boundaries.txt"
MODEL = "gpt-6-luna"
LIMIT = 30_000_000
GENERATION_CHECKS = '''提交前逐项检查P/F的业务逻辑，不输出检查过程：
1. F必须针对本次目标的实际结果；已纠正的失败尝试、额外无害记录、另一对象的错误，不能否定一个满足存在性要求的正确对象。
2. 条件不触发时允许正确的不操作分支；不能在F中无条件写“未执行即失败”。条件未知和条件已知为假分开。
3. “最早/最高”只约束该业务值；接口的ID二级排序不是题面授权的强制并列决胜规则。
4. retrieve/read不自动变成最终展示全部字段。只把题目要求报告、交付的内容列为交付义务。
5. 对一项具体必要违例，F不要求先知道其他无关字段、对象或完整集合。证明不存在/漏项所必需的完整范围证据仍须保留。
6. 允许的等价路径必须同时在P和F成立；若明确已知交付值错误或缺失，其他单位/身份歧义不得覆盖独立违例。'''


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("frozen_content_changed:" + str(path))
    else:
        write_json(path, value)
    return value


def config():
    expanded = read_json(ROOT / "configs/harness/v2-expanded.json")
    api = dict(base_url="https://api.openai.com/v1", model=MODEL,
               api_key_env="OPENAI_API_KEY", context_limit=128000, timeout=600)
    return RunConfig.model_validate({**expanded, "actor": "api", "judge": "api",
        "judge_protocol": "rubric_v2", "purpose": "development", "isolation": "strict",
        "actor_api": {**api, "reasoning_effort": "none"},
        "judge_api": {**api, "reasoning_effort": "low"},
        "workers": 3, "model_concurrency": 3, "judge_concurrency": 3,
        "max_physical_requests": 12000, "max_total_tokens": LIMIT})


def prepare(run, data):
    run, data = Path(run).resolve(), Path(data).resolve()
    if run.exists() or data.exists():
        raise ValueError("prepare_requires_new_directories")
    reference = ROOT / "data/awm-dde80a0/gen_scenario.jsonl"
    sources = [Path(__file__), BOUNDARIES, Path(prompts.__file__), *sorted((ROOT / "configs/rubric/d1").glob("*.txt")),
               *sorted((ROOT / "harness_awm").glob("*.py")),
               ROOT / "scripts/rubric_d1.py", ROOT / "scripts/rubric_candidates_100.py",
               ROOT / "scripts/rubric_candidates.py", ROOT / "scripts/rubric_pilot.py",
               *[ROOT / 'agent-world-model/awm/core' / (n+'.py') for n in ['db','sample','server']]]
    run.mkdir(parents=True)
    data.mkdir(parents=True)
    manifest = dict(schema_version="awm-new-pilot-v1", data=str(data), reference=str(reference),
        reference_sha256=file_hash(reference), config=config().model_dump(), model=MODEL,
        max_total_tokens=LIMIT, max_requests=12000, scenarios=10, tasks_per_scenario=10,
        first_stage_scenarios=2, max_repairs=1, split="development", seed=20261104,
        implementation_hashes={str(p): file_hash(p) for p in sources},
        historical_ledger_sha256=file_hash(ROOT / "runs/rubric_luna_registry/batches.json"),
        authorization="User approved 10x10, staged 2x10 gate, OpenAI API, 30000000 total tokens.")
    freeze(run / "manifest.json", manifest)
    freeze(run / "manifest.sha256.json", {"sha256": file_hash(run / "manifest.json")})
    return {"prepared": str(run), "data": str(data), "paid_requests": 0}


def load_manifest(run, verify_code=True):
    run = Path(run)
    if file_hash(run / "manifest.json") != read_json(run / "manifest.sha256.json")["sha256"]:
        raise ValueError("manifest_changed")
    m = read_json(run / "manifest.json")
    if file_hash(m["reference"]) != m["reference_sha256"]:
        raise ValueError("reference_changed")
    hashes=m['implementation_hashes']
    parent=file_hash(run/'manifest.json')
    for revision in sorted((run/'execution_revisions').glob('*.json')):
        update=read_json(revision)
        if update['parent_sha256']!=parent:
            raise ValueError('execution_revision_chain_broken')
        hashes=update['implementation_hashes']
        parent=file_hash(revision)
    if verify_code:
        for path, expected in hashes.items():
            if file_hash(path) != expected:
                raise ValueError("implementation_changed:" + path)
    return m


def original_input(task_id, task, spec, schema):
    return dict(task_id=task_id, task=task, quality_policy=None,
        sources=[dict(id="task", kind="task", text=task),
                 dict(id="contract", kind="public_contract", text=canonical(spec)),
                 *[dict(id="schema:" + t["name"], kind="schema",
                        text=t["ddl"] + "\n" + "\n".join(t.get("indexes", [])))
                   for t in schema["tables"]]],
        available_evidence_types=["trajectory", "initial_database", "final_database", "tool_results"],
        evidence_note="Runtime evidence is collected after generation. No runtime data is supplied here.")


def compatible_verifier(schema):
    # SQL-mode evidence collector, not a success oracle. V2 does not execute it.
    names = [t["name"] for t in schema["tables"]]
    return "\n".join([
        "def verify_task(initial_db_path, final_db_path):", "    import sqlite3",
        "    from pathlib import Path", "    result = {}",
        "    for label, path in [('initial', initial_db_path), ('final', final_db_path)]:",
        "        with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:",
        "            db.row_factory = sqlite3.Row", "            result[label] = {}",
        "            for name in " + repr(names) + ":",
        "                quoted = '\"' + name.replace('\"', '\"\"') + '\"'",
        "                result[label][name] = [dict(r) for r in db.execute('SELECT * FROM ' + quoted)]",
        "    return result", ""])


def normalize_engine_layout(code):
    """AWM's server wrapper rewrites create_engine by line, not by Python AST.

    Collapse only multi-line assignment statements containing create_engine;
    prove AST equality so this compatibility fix cannot change business logic.
    Invalid generated syntax is left for the existing bounded repair path.
    """
    try:
        tree=ast.parse(code)
    except SyntaxError:
        return code
    lines=code.splitlines(keepends=True)
    edits=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AnnAssign)) and node.end_lineno>node.lineno and any(
            isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='create_engine'
            for n in ast.walk(node)):
            edits.append((node.lineno-1,node.end_lineno,' '*node.col_offset+ast.unparse(node)+'\n'))
    for start,end,replacement in sorted(edits,reverse=True):
        lines[start:end]=[replacement]
    fixed=''.join(lines)
    if ast.dump(ast.parse(fixed))!=ast.dump(tree):
        raise ValueError('engine_layout_changed_semantics')
    return fixed


def task_spec(m, folder, index, instruction):
    folder = Path(folder)
    initial, env, verifier = folder / "initial.db", folder / "environment.json", folder / "verifiers" / f"{index:02d}.json"
    db_check(initial)
    scenario = read_json(env)["scenario"]
    v = read_json(verifier)
    if v["scenario"] != scenario or v["task_idx"] != index or v["task"] != instruction:
        raise ValueError("task_verifier_binding_mismatch")
    return TaskSpec(protocol_id="awm-single-json-v2", dataset_version=Path(m["data"]).name,
        split="development", lineage_group_id=scenario, scenario=scenario, native_task_id=index,
        task_instance_id=f"{Path(m['data']).name}:{scenario}:{index}", instruction=instruction,
        initial_state_ref=str(initial), initial_state_hash=file_hash(initial),
        initial_state_fingerprint=db_fingerprint(initial), environment_ref=str(env),
        environment_code_hash=digest(read_json(env)), verifier_ref=str(verifier), verifier_hash=digest(v),
        source_manifest_hash=digest(m), cleaning_status="new_pilot_development",
        readiness_profile={"runtime_ready": True, "reward_ready": False})


def gate(worlds):
    cases = [t for w in worlds for t in w.get("tasks", [])]
    evaluated = [t["evaluation"] for t in cases if t.get("evaluation", {}).get("evaluation_status") == "valid"]
    unknown = sum(e["business_label"] == "unknown" for e in evaluated)
    reasons = []
    if len(worlds) != 2 or not all(w.get("environment_passed") for w in worlds):
        reasons.append("environment_acceptance_incomplete")
    if len(cases) != 20 or not all(t.get("rubric_status") == "passed" for t in cases):
        reasons.append("rubric_acceptance_incomplete")
    if len(evaluated) < 18:
        reasons.append("fewer_than_18_valid_evaluations")
    if evaluated and unknown / len(evaluated) > .1:
        reasons.append("unknown_above_10_percent")
    if not all(w.get("controls_passed") for w in worlds):
        reasons.append("control_acceptance_incomplete")
    if not all(w.get("audit_passed") for w in worlds):
        reasons.append("source_and_evidence_audit_incomplete")
    return {"passed": not reasons, "reasons": reasons, "valid": len(evaluated), "unknown": unknown}


class Pilot:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.m = load_manifest(self.run)
        self.data = Path(self.m["data"])
        self.config = RunConfig.model_validate(self.m["config"])
        self.budget = Budget(self.run / "budget.sqlite", self.m["max_requests"], self.m["max_total_tokens"])
        self.sem, self.env_sem = asyncio.Semaphore(3), asyncio.Semaphore(2)
        self.gen = self.config.judge_api.model_copy(update={"reasoning_effort": "high"})

    async def request(self, key, messages, *, code=False, limit=18000):
        if not code and not any('json' in m.get('content','').lower() for m in messages):
            # OpenAI JSON mode requires an explicit instruction. A different key
            # retains the rejected pre-fix request and its unknown-usage reserve.
            messages=[dict(m) for m in messages]
            messages[0]['content']+=' Return JSON only.'
            key+='/json_contract_v1'
        dest = self.run / "requests" / key
        if (dest/'identity.json').exists():
            # Preserve the exact settings of earlier requests on resume.
            limit=read_json(dest/'identity.json')['limit']
        elif limit<32768 and fits_context(messages,32768,128000):
            # Measured high-effort calls sometimes use >14k tokens before any
            # visible output. Reserve sufficient room on fresh calls, not retries.
            limit=32768
        identity = dict(messages=messages, model=MODEL, effort="high", code=code, limit=limit)
        if not fits_context(messages,limit,128000):
            # Preserve rejected preflight identities. No paid attempt may be
            # replayed, and the complete source remains unchanged.
            available=128000-estimate_tokens(messages)-512
            if available<=0 or self.budget.request_record(key) or (dest/'response.json').exists():
                raise ValueError('context_repair_not_safe:'+key)
            return await self.request(key+'/context_fit_v1',messages,code=code,limit=available)
        freeze(dest / "identity.json", identity)
        response = dest / "response.json"
        if response.exists():
            raw = read_json(response)
        else:
            if self.budget.request_record(key):
                raise ValueError("interrupted_request_not_replayed:" + key)
            async with self.sem:
                print(canonical({"request": key}), flush=True)
                raw = await api_request(self.gen, dict(model=MODEL, messages=messages,
                    max_completion_tokens=limit, temperature=1,
                    **({} if code else {"response_format": {"type": "json_object"}})),
                    self.budget, key, key.split('/')[0], dest)
        choice = raw["choices"][0]
        if choice["finish_reason"] != "stop":
            if choice['finish_reason']=='length' and limit<32768 and '/completion_repair' not in key and ('/r1' not in key or key.startswith('rubric_review/')):
                # One bounded correction of a proven output-budget failure. Keep
                # the incomplete response invalid and retain its actual cost.
                corrected=[dict(m) for m in messages]
                corrected.append({'role':'user','content':'The previous attempt exhausted its output-token allowance before producing a complete response. Return one complete result for the same requirements; be concise without omitting obligations. The output allowance is now larger.'})
                return await self.request(key+'/completion_repair',corrected,code=code,limit=32768)
            raise ValueError("incomplete_generation:" + key)
        content = choice["message"].get("content") or ""
        return content if code else d1.base.json_spans(content)[0]

    async def embeddings(self, records):
        import numpy as np
        from openai import AsyncOpenAI
        output = []
        for offset in range(0, len(records), 50):
            texts = [r["name"] + ": " + r["description"] for r in records[offset:offset+50]]
            key = "embedding/" + digest(texts)
            dest = self.run / "requests" / key
            freeze(dest / "identity.json", {"input": texts, "model": "text-embedding-3-large"})
            if (dest / "response.json").exists():
                data = read_json(dest / "response.json")
            else:
                if self.budget.request_record(key):
                    raise ValueError("interrupted_embedding_not_replayed")
                async with self.sem:
                    rid = self.budget.reserve(key, "embedding", [{"role": "user", "content": canonical(texts)}], 0)
                    try:
                        async with AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], max_retries=0, timeout=180) as client:
                            response = await client.embeddings.create(model="text-embedding-3-large", input=texts)
                        data = response.model_dump(mode="json")
                        write_json(dest / "response.json", data)
                        self.budget.finish(rid, data["usage"])
                    except BaseException:
                        self.budget.finish(rid)
                        raise
            output.extend(r["embedding"] for r in sorted(data["data"], key=lambda r:r["index"]))
        matrix = np.array(output)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    async def scenarios(self):
        target = self.data / "scenarios.json"
        if target.exists():
            return read_json(target)
        reference = rows(self.m["reference"])
        old_names = {normalize_scenario_name(r["name"]) for r in reference}
        # Existing names plus nearest full descriptions provide two complementary checks.
        prompt = {"existing_scenario_names": sorted(old_names), "count": 10,
            "requirements": "Generate ten genuinely new specialized SQLite/API-simulatable business workflows in at least five different domains. Avoid generic renamed CRM, shopping, booking or todo worlds. Each world must support retrieval, mutation, multistep dependency, conditional and set-wide constraints. No external services, credentials, human approvals or real payments. Bounded pilot: 3-6 tables, 6-12 endpoints, deterministic initial state. Return scenarios [{name: unique snake_case, description: detailed workflow, category: domain}]."}
        previous = None
        old_vectors = None
        for revision in range(2):
            messages = [{"role": "system", "content": "Design diverse agent benchmark worlds. Output JSON only."},
                        {"role": "user", "content": canonical(prompt)}]
            if previous:
                messages.append({"role": "user", "content": canonical(previous)})
            result = await self.request(f"scenario/r{revision}", messages, limit=12000)
            candidates = result["scenarios"]
            names = [normalize_scenario_name(r["name"]) for r in candidates]
            if len(names) != 10 or len(set(names)) != 10 or any(n in old_names or not n for n in names) or len({r['category'] for r in candidates}) < 5:
                previous = {"previous": result, "repair": "Need exactly 10 unique NEW names and at least 5 categories."}
                continue
            if old_vectors is None:
                old_vectors = await self.embeddings(reference)
            new_vectors = await self.embeddings(candidates)
            similarity = new_vectors @ old_vectors.T
            nearest = []
            for i, row in enumerate(candidates):
                indexes = similarity[i].argsort()[-3:][::-1]
                nearest.append({"candidate": row, "matches": [{"reference": reference[int(j)], "similarity": float(similarity[i,j])} for j in indexes]})
            review = await self.request(f"scenario_review/r{revision}", [
                {"role":"system", "content":"Independently compare business workflows, not just names. Return {distinct:boolean, issues:[specific duplicated workflows]}. Reject near-renamed old scenarios or duplicates within candidates."},
                {"role":"user", "content":canonical(nearest)}], limit=8000)
            max_cross = float(similarity.max())
            import numpy as np
            within = new_vectors @ new_vectors.T
            np.fill_diagonal(within, 0)
            if review.get('distinct') is True and max_cross < .85 and float(within.max()) < .85:
                accepted = [{**r, "name":n} for r,n in zip(candidates,names)]
                freeze(self.data / "scenario_dedup.json", {"nearest":nearest, "review":review,
                    "maximum_cross_similarity":max_cross, "maximum_within_similarity":float(within.max())})
                return freeze(target, accepted)
            previous = {"previous":result, "review":review, "nearest":nearest,
                        "repair":"Replace duplicated workflows; similarity must be below .85."}
        raise ValueError("scenario_diversity_gate_failed")

    async def generate_stage(self, world, stage, fields, *, code=False, extra="", previous=None):
        system = getattr(prompts, stage + "_SYSTEM_PROMPT")
        user = getattr(prompts, stage + "_USER_PROMPT").format(**fields)
        messages = [{"role":"system", "content":system}, {"role":"user", "content":user + "\n" + extra}]
        if previous:
            messages.append({"role":"user", "content":canonical(previous)})
        return await self.request(f"assets/{world}/{stage.lower()}/r{int(previous is not None)}", messages,
                                  code=code, limit=22000 if code else 14000)

    async def build(self, scenario):
        name = scenario["name"]
        folder = self.data / "worlds" / name
        if (folder / "ready.json").exists():
            ready = read_json(folder / "ready.json")
            for rel, expected in ready["hashes"].items():
                if file_hash(folder / rel) != expected:
                    raise ValueError("world_asset_changed:" + rel)
            return folder
        async with self.env_sem:
            tasks = await self.generate_stage(name, "TASK_GENERATION", dict(num_tasks=10,
                scenario_name=name, scenario_description=scenario['description']), extra=
                "Return exactly 10 tasks in {tasks:[strings]}. English tasks. Explicit units and absolute dates with years; never today/next week or unspecified current user. Include lookup, mutation, multistep dependencies, conditional branches and complete-set obligations across the ten. First task must have at least two independently verifiable persistent changes (so partial completion can be tested). Each task starts independently from the same database. 3-6 tables and 6-12 API endpoints must suffice. Do not simplify away meaningful business constraints.")
            if len(tasks.get('tasks',[])) != 10 or not all(isinstance(t,str) and t.strip() for t in tasks['tasks']):
                raise ValueError("invalid_task_count_or_type")
            freeze(folder / "tasks.json", tasks)
            joined = "\n".join(f"- {t}" for t in tasks['tasks'])
            schema = await self.generate_stage(name, "DATABASE_GENERATION", dict(num_tasks=10,
                scenario_name=name, user_intentions=joined), extra="Use 3-6 tables. No example INSERTs in schema: examples must be empty arrays. Explicit stable primary keys. No triggers using wall clock; use deterministic values in sample data.")
            sample = await self.generate_stage(name, "SAMPLE_DATA_GENERATION", dict(scenario_name=name,
                tasks_list=joined, database_schema=format_db_schema(schema)), extra=
                "Use deterministic literal dates and IDs, no datetime('now'). Seed all prerequisites for every task. Do not pre-complete tasks requesting new records or updates. No duplicate example inserts.")
            for rev in range(2):
                freeze(folder / f"schema.r{rev}.json", schema)
                freeze(folder / f"sample.r{rev}.json", sample)
                init = await worker(self.config, "initialize", folder / f"initialize.r{rev}",
                                    dict(scenario=name, schema=schema, sample=sample))
                try:
                    if not init['complete']:
                        raise ValueError(canonical(init))
                    db_check(init['path'])
                    break
                except ValueError as exc:
                    if rev:
                        raise
                    repaired = await self.request(f"asset_repair/{name}/database", [
                        {"role":"system", "content":"Repair SQLite schema/sample initialization defects without changing task requirements. Return {schema: original schema format, sample: original sample format}."},
                        {"role":"user", "content":canonical(dict(tasks=tasks,schema=schema,sample=sample,error=str(exc)))}])
                    schema,sample = repaired['schema'],repaired['sample']
            freeze(folder / "schema.json", schema)
            freeze(folder / "sample.json", sample)
            if not (folder / "initial.db").exists():
                snapshot(init['path'], folder / "initial.db")
                (folder / "initial.db").chmod(0o444)
            spec = await self.generate_stage(name, "API_GENERATION", dict(scenario_name=name,
                tasks_list=joined, database_schema=format_db_schema(schema)), extra=
                "Use 6-12 endpoints supporting all ten tasks. Explicit units, filters, ordering, pagination, identity and update semantics. Include complete listing and direct reads for checking writes. No authentication/external side effects.")
            freeze(folder / "spec.json", spec)
            fields = dict(scenario_name=name, database_schema=format_db_schema(schema),
                          api_spec=canonical(spec), PYTHON_VERSION="3.12")
            code = await self.generate_stage(name, "ENVIRONMENT_GENERATION", fields, code=True,
                extra="Preserve ALL public contract semantics. One-line engine = create_engine(...) assignment. Include exactly the standard if __name__ == '__main__': uvicorn.run(app, ...) entrypoint so AWM can attach MCP. No network calls or subprocesses. Do not seed data on startup; supplied database is already initialized.")
            for i, instruction in enumerate(tasks['tasks']):
                freeze(folder / "verifiers" / f"{i:02d}.json", dict(scenario=name, task_idx=i, task=instruction,
                    verification={"code":compatible_verifier(schema), "raw_response":"deterministic full-state SQL evidence collector; not a success classifier"}))
            for rev in range(2):
                code=normalize_engine_layout(code)
                env = dict(scenario=name, db_path=str(folder/'initial.db'), full_code=code)
                freeze(folder / f"environment.r{rev}.json", env)
                # Candidate path is versioned. Only accepted code is installed at environment.json.
                candidate = task_spec_candidate(self.m, folder, tasks['tasks'][0], env, rev)
                adapter = AWMEnvAdapter(candidate, self.config, folder / f"smoke.r{rev}")
                try:
                    compile(code, '<generated-env>', 'exec')
                    await adapter.prepare()
                    if not adapter.tools:
                        raise ValueError("empty_tool_catalog")
                    freeze(folder / "tool_catalog.json", adapter.tools)
                    break
                except Exception as exc:
                    if rev:
                        raise
                    log = folder / f"smoke.r{rev}" / "server.log"
                    code = await self.request(f"asset_repair/{name}/environment_runtime/r1", [
                        {'role':'system','content':'Repair only the demonstrated Python runtime/startup defect. Preserve every endpoint, parameter, database operation and business rule. Return the complete Python module, no prose or fences. Keep create_engine assignments on one line and the standard uvicorn.run(app, ...) main entrypoint.'},
                        {'role':'user','content':canonical({'previous_code':code,'schema':schema,'error':str(exc),
                            'server_log':log.read_text()[-8000:] if log.exists() else ''})}],code=True,limit=22000)
                finally:
                    await adapter.close()
            freeze(folder / "environment.json", env)
            paths = [folder / x for x in ['tasks.json','schema.json','sample.json','spec.json','initial.db','environment.json','tool_catalog.json']]
            paths += sorted((folder / 'verifiers').glob('*.json'))
            freeze(folder / "ready.json", dict(hashes={str(p.relative_to(folder)):file_hash(p) for p in paths},
                initialization=init, startup=True, environment_acceptance="pending_real_write_reset_controls"))
            return folder

    async def rubric(self, folder, index):
        name = folder.name
        path = folder / "rubrics" / f"{index:02d}"
        if (path / "result.json").exists():
            return read_json(path / "result.json")
        task = read_json(folder / "tasks.json")['tasks'][index]
        inp = original_input(f"{name}:{index}", task, read_json(folder/'spec.json'), read_json(folder/'schema.json'))
        freeze(path / "original.json", inp)
        previous, feedback = None, None
        for revision in range(2):
            try:
                request = d1.generation_request(inp, previous, feedback)
                request['messages'][0]['content'] += '\n' + BOUNDARIES.read_text() + '\n' + GENERATION_CHECKS
                if feedback and previous is None:
                    request['messages'].append({'role':'user','content':canonical({'revision_feedback':feedback,
                        'instruction':'The previous output could not be parsed safely. Generate a complete valid JSON result, with no duplicate keys, preserving all original obligations.'})})
                selected = await self.request(f"rubric/{name}/{index:02d}/r{revision}", request['messages'], limit=18000)
                freeze(path / f"selection.r{revision}.json", selected)
                _, pack = d1.base.prepare_input(inp)
                compiled = d1.compile_rubric(selected, inp, pack)
                request = d1.review_request(inp, selected)
                request['messages'][0]['content'] += '\n' + BOUNDARIES.read_text()
                review = await self.request(f"rubric_review/{name}/{index:02d}/r{revision}", request['messages'], limit=12000)
                validation = d1.base.evaluate(review, inp['task_id'], {r['id'] for r in compiled['rubric']},
                                               {r['candidate_id'] for r in pack['candidates']})
                freeze(path / f"review.r{revision}.json", dict(response=review, validation=validation))
                if validation['status'] == 'passed':
                    freeze(path / "rubric.json", compiled)
                    return freeze(path / "result.json", dict(status='passed', revision=revision,
                        scores=validation['scores'], unresolved=len(compiled['unresolved'])))
                previous,feedback = selected,dict(review=review,validation=validation)
            except (ValueError, KeyError, TypeError) as exc:
                previous = locals().get('selected')
                feedback = {"validation_error":str(exc)}
        return freeze(path / "result.json", dict(status='rejected', feedback=feedback))

    async def episode(self, task, original, rubric, path, actions=None):
        path = Path(path)
        identity = dict(task=task.model_dump(), original=digest(original), rubric=digest(rubric),
                        actions=actions, config=digest(self.config.model_dump()))
        freeze(path / 'identity.json', identity)
        if (path/'result.json').exists():
            return read_json(path/'result.json')
        ep = path/'episode'
        if ep.exists() and not (ep/'episode_result.json').exists():
            return freeze(path/'result.json', dict(error='interrupted_episode_not_replayed'))
        if (ep/'episode_result.json').exists():
            execution = read_json(ep/'episode_result.json')
        else:
            backend = APIBackend(self.config,self.budget,self.sem,str(path.relative_to(self.run))) if actions is None else ScriptedBackend(actions)
            execution = await EpisodeRunner(self.config,self.budget,self.sem).run(
                task,ep,str(path.relative_to(self.run)),backend,evaluate_after=False)
        result = dict(execution=execution)
        if execution['terminal_state_status']=='trusted':
            bundle = prepare_input(ep,original,rubric)
            freeze(path/'judge_input.json',bundle)
            result['evaluation'] = await evaluate_input(bundle,self.config,self.budget,self.sem,path/'evaluations')
        return freeze(path/'result.json',result)

    async def controls(self, folder):
        # Source-derived control oracle is generated without rubric or judge labels.
        dest = self.run/'worlds'/folder.name/'controls'
        if (dest/'result.json').exists():
            return read_json(dest/'result.json')
        task = task_spec(self.m,folder,0,read_json(folder/'tasks.json')['tasks'][0])
        inp = read_json(folder/'rubrics/00/original.json')
        rubric = read_json(folder/'rubrics/00/rubric.json')
        control = await self.request(f"controls/{folder.name}/plan", [
            {"role":"system", "content": "Design executable test controls from the task, tool schemas and initial data. Do not change the task. Return {complete:{calls:[{name,arguments}],answer:string}, partial:{calls:[{name,arguments}],answer:string}, assertions:[{sql:read-only SELECT, initial:array-of-rows, complete:array-of-rows, partial:array-of-rows, obligation:string}]}. The complete control must satisfy every task obligation. Partial must actually persist one requested change but omit another. Use known initial IDs; avoid unknown new IDs (resolve via stable unique values or deterministic documented IDs). All SQL results deterministic ORDER BY. Assertions must separately cover each requested persistent effect and distinguish initial, partial and complete. At least two assertions. No fabrication of runtime responses."},
            {"role":"user", "content":canonical(dict(task=task.instruction,schema=read_json(folder/'schema.json'),
                sample=read_json(folder/'sample.json'),tools=read_json(folder/'tool_catalog.json')))}], limit=12000)
        freeze(dest/'plan.json',control)
        if len(control.get('assertions',[]))<2:
            return freeze(dest/'result.json',dict(passed=False,error='insufficient_control_assertions'))
        results = {}
        for mode in ['complete','partial','unexecuted']:
            actions = []
            for c in control[mode]['calls'] if mode!='unexecuted' else []:
                actions += [dict(type='get_tool_schema',name=c['name']),dict(type='call_tool',**c)]
            answer = control[mode]['answer'] if mode!='unexecuted' else 'I have not executed any requested changes.'
            actions.append(dict(type='finish',status='completed' if mode=='complete' else 'incomplete',
                answer=answer, **({} if mode=='complete' else {'reason':'Deliberate negative control.'})))
            if mode=='unexecuted':
                actions.append(dict(actions[-1]))  # V2 confirms an early zero-call finish once.
            result = await self.episode(task,inp,rubric,dest/mode,actions)
            try:
                ep = dest/mode/'episode'
                checks = []
                for assertion in control['assertions']:
                    initial,initial_ok = assertion_query(ep/'initial.db',assertion['sql'],assertion['initial'])
                    final,final_ok = assertion_query(ep/'final.db',assertion['sql'],assertion['initial' if mode=='unexecuted' else mode])
                    checks.append(dict(obligation=assertion['obligation'],initial=initial,final=final,
                        passed=initial_ok and final_ok))
                expected = 'pass' if mode=='complete' else 'fail'
                e = result.get('evaluation',{})
                changed = db_fingerprint(ep/'initial.db') != db_fingerprint(ep/'final.db')
                results[mode] = dict(checks=checks, business_label=e.get('business_label'),
                    passed=all(c['passed'] for c in checks) and e.get('evaluation_status')=='valid' and
                    e.get('business_label')==expected and (changed if mode!='unexecuted' else not changed))
            except (ValueError,OSError,sqlite3.Error) as exc:
                results[mode] = dict(passed=False,error=str(exc))
        return freeze(dest/'result.json',dict(passed=all(r['passed'] for r in results.values()),cases=results))

    async def world(self, scenario):
        name = scenario['name']
        output = self.run/'worlds'/name/'result.json'
        if output.exists():
            return read_json(output)
        result = dict(scenario=name,tasks=[],environment_passed=False,controls_passed=False,audit_passed=False)
        try:
            folder = await self.build(scenario)
            # Bound generation queue; no separate worker pool can exceed global API concurrency.
            reviews = await asyncio.gather(*(self.rubric(folder,i) for i in range(10)))
            tasks = read_json(folder/'tasks.json')['tasks']
            specs = [task_spec(self.m,folder,i,t) for i,t in enumerate(tasks)]
            registry = folder/'registry'
            registry.mkdir(exist_ok=True)
            text = ''.join(canonical(t.model_dump())+'\n' for t in specs)
            if (registry/'tasks.jsonl').exists() and (registry/'tasks.jsonl').read_text()!=text:
                raise ValueError('registry_changed')
            (registry/'tasks.jsonl').write_text(text)
            freeze(registry/'registry_manifest.json', dict(schema_version='awm-registry-v1',
                files={'tasks.jsonl':file_hash(registry/'tasks.jsonl')},task_count=10,development_only=True))
            load_registry(registry)
            mapping = {}
            for i,r in enumerate(reviews):
                if r['status']=='passed':
                    mapping[specs[i].task_instance_id] = {k:dict(path=str(folder/'rubrics'/f'{i:02d}'/(k+'.json')),
                        sha256=file_hash(folder/'rubrics'/f'{i:02d}'/(k+'.json'))) for k in ['original','rubric']}
            manifest_value=dict(schema_version='awm-judge-input-manifest-v2',tasks=mapping)
            manifest_path=folder/'judge_inputs_manifest.json'
            if manifest_path.exists() and read_json(manifest_path)!=manifest_value:
                manifest_path=folder/f'judge_inputs_manifest.{digest(manifest_value)[:16]}.json'
            freeze(manifest_path,manifest_value)
            result['judge_inputs_manifest']=str(manifest_path)
            async def one(i):
                r = dict(index=i,task=tasks[i],rubric_status=reviews[i]['status'],rubric_review=reviews[i])
                if reviews[i]['status']=='passed':
                    p=folder/'rubrics'/f'{i:02d}'
                    r.update(await self.episode(specs[i],read_json(p/'original.json'),read_json(p/'rubric.json'),
                        self.run/'worlds'/name/'tasks'/f'{i:02d}'))
                return r
            # Three active episodes limit runtime processes as well as model calls.
            for start in range(0,10,3):
                result['tasks'].extend(await asyncio.gather(*(one(i) for i in range(start,min(start+3,10)))))
                write_json(output.with_name('progress.json'),result)
            if reviews[0]['status']=='passed':
                controls = await self.controls(folder)
                result['controls_passed'] = controls['passed']
                result['environment_passed'] = controls['passed']
            result['audit_required'] = [r['index'] for r in result['tasks'] if
                r.get('evaluation',{}).get('evaluation_status')!='valid' or
                r.get('evaluation',{}).get('business_label')=='unknown' or
                1 in r['rubric_review'].get('scores',{}).values()]
            # The coordinator inspects these cases and writes a versioned evidence audit.
            audit_path = self.run/'worlds'/name/'audit.json'
            result['audit_passed'] = not result['audit_required']
            if audit_path.exists():
                audit=read_json(audit_path)
                result['audit_passed'] = audit.get('passed') is True and set(audit.get('reviewed_indexes',[]))>=set(result['audit_required'])
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        freeze(output,result)
        print(canonical(dict(world=name,tasks=len(result['tasks']),controls=result['controls_passed'],error=result.get('error'))),flush=True)
        return result

    async def run_all(self, phase='all'):
        audited=self.run/'pilot_gate.audited.json'
        if audited.exists() and not read_json(audited)['passed']:
            return report(self.run)  # A completed failed gate must not dispatch again.
        scenarios = await self.scenarios()
        if phase=='scenarios':
            return report(self.run)
        if phase in ('assets','rubrics'):
            for scenario in scenarios[:2]:
                folder=await self.build(scenario)
                if phase=='rubrics':
                    await asyncio.gather(*(self.rubric(folder,i) for i in range(10)))
            return report(self.run)
        first=await asyncio.gather(*(self.world(scenario) for scenario in scenarios[:2]))
        report(self.run)
        decision=gate(first)
        freeze(self.run/'pilot_gate.json',decision)
        if decision['passed'] and phase=='all':
            for start in range(2,10,2):
                await asyncio.gather(*(self.world(scenario) for scenario in scenarios[start:start+2]))
                report(self.run)
        return report(self.run)


def task_spec_candidate(m,folder,instruction,env,revision):
    # Smoke initialization uses the same TaskSpec invariants as the final adapter.
    return TaskSpec(protocol_id='awm-single-json-v2',dataset_version=Path(m['data']).name,split='development',
        lineage_group_id=folder.name,scenario=folder.name,native_task_id=0,
        task_instance_id=f"smoke:{folder.name}:r{revision}",instruction=instruction,
        initial_state_ref=str(folder/'initial.db'),initial_state_hash=file_hash(folder/'initial.db'),
        initial_state_fingerprint=db_fingerprint(folder/'initial.db'),
        environment_ref=str(folder/f'environment.r{revision}.json'),environment_code_hash=digest(env),
        verifier_ref=str(folder/'verifiers/00.json'),verifier_hash=digest(read_json(folder/'verifiers/00.json')),
        source_manifest_hash=digest(m),readiness_profile={'runtime_ready':True,'reward_ready':False})


def query(path,sql):
    from harness_awm.judge_evidence import EvidenceStore
    # Reuse the Judge's hardened read-only executor rather than accepting arbitrary SQL.
    store=EvidenceStore(path,path,seconds=3,max_rows=1000)
    with store.connect('initial') as db:
        values=db.execute(sql).fetchmany(1001)
        if len(values)>1000:
            raise ValueError('control_assertion_truncated')
        return [list(r) for r in values]


def assertion_query(path,sql,expected):
    """Compare positional or named JSON rows against actual SQL column order."""
    from harness_awm.judge_evidence import EvidenceStore
    store=EvidenceStore(path,path,seconds=3,max_rows=1000)
    with store.connect('initial') as db:
        cur=db.execute(sql)
        columns=[c[0] for c in cur.description]
        actual=[list(r) for r in cur.fetchmany(1001)]
    if len(actual)>1000 or not isinstance(expected,list):
        raise ValueError('invalid_or_truncated_control_assertion')
    normalized=[]
    for row in expected:
        if isinstance(row,dict):
            if len(set(columns))!=len(columns) or set(row)!=set(columns):
                raise ValueError('control_assertion_column_mismatch')
            normalized.append([row[c] for c in columns])
        elif isinstance(row,list) and len(row)==len(columns):
            normalized.append(row)
        else:
            raise ValueError('invalid_control_assertion_row')
    return actual,actual==normalized


def report(run):
    run=Path(run)
    m=load_manifest(run,False)
    worlds=[read_json(p.with_name('result.audited.json') if p.with_name('result.audited.json').exists() else p)
            for p in sorted((run/'worlds').glob('*/result.json'))]
    tasks=[{**t,'scenario':w['scenario']} for w in worlds for t in w.get('tasks',[])]
    evaluations=[t['evaluation'] for t in tasks if 'evaluation' in t]
    budget=Budget(run/'budget.sqlite',m['max_requests'],m['max_total_tokens'])
    with sqlite3.connect(run/'budget.sqlite') as db:
        stages=[dict(stage=s,requests=n,tokens_accounted_or_reserved=t) for s,n,t in
                db.execute('SELECT stage,count(*),sum(reserved) FROM requests GROUP BY stage')]
    result=dict(planned_scenarios=10,planned_tasks=100,worlds_recorded=len(worlds),
        generated_tasks=sum(len(read_json(p)['tasks']) for p in Path(m['data']).glob('worlds/*/tasks.json')),
        rubric_passed=sum(t['rubric_status']=='passed' for t in tasks),executed=len(evaluations),
        environment_passed=sum(w.get('environment_passed',False) for w in worlds),
        evaluation_statuses=dict(Counter(e['evaluation_status'] for e in evaluations)),
        task_statuses=dict(Counter(e['task_status'] for e in evaluations)),
        budget=budget.summary(),stage_usage=stages,
        gate=read_json(run/'pilot_gate.audited.json') if (run/'pilot_gate.audited.json').exists() else
            (read_json(run/'pilot_gate.json') if (run/'pilot_gate.json').exists() else None),
        worlds=worlds,independent_accuracy_measured=False,reward_exported=False)
    write_json(run/'summary.json',result)
    lines=['# AWM 新场景开发试跑', '',f"目标10场景/100题；已生成{result['generated_tasks']}题，rubric通过{result['rubric_passed']}题，已记录判分{result['executed']}题。",
        f"环境验收通过{result['environment_passed']}个。任务状态：{result['task_statuses']}。评测状态：{result['evaluation_statuses']}。",
        f"预算已用或预留：{result['budget']['tokens_accounted_or_reserved']:,}/{LIMIT:,} tokens。",
        f"首批续跑门槛：{result['gate']}。",'',
        '模型判分不是人工金标准；暂定通过不计为确认成功。本批只用于开发，未导出GRPO奖励。','',
        '| 场景 | 题号 | rubric | 判分状态 | 业务结果 |','|---|---:|---|---|---|']
    for t in tasks:
        e=t.get('evaluation',{})
        lines.append(f"| {t['scenario']} | {t['index']} | {t['rubric_status']} | {e.get('evaluation_status','not_run')} | {e.get('business_label','—')} |")
    lines+=['','## 阻塞与错误']+[f"- {w['scenario']}: {w.get('error') or w.get('audit_required',[])}" for w in worlds if w.get('error') or w.get('audit_required')]
    (run/'report_zh.md').write_text('\n'.join(lines)+'\n')
    return {k:v for k,v in result.items() if k!='worlds'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','repair','report'])
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--phase',choices=['scenarios','assets','rubrics','pilot','all'],default='all')
    a=p.parse_args()
    if a.action=='prepare':
        result=prepare(a.run,a.data)
    elif a.action=='report':
        result=report(a.run)
    else:
        from dotenv import dotenv_values
        if not os.environ.get('OPENAI_API_KEY'):
            key=dotenv_values(ROOT/'.env',interpolate=False).get('OPENAI_API_KEY')
            if not key:
                raise ValueError('OPENAI_API_KEY_missing')
            os.environ['OPENAI_API_KEY']=key
        with (a.run/'runner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if a.action=='repair':
                # Only output-truncated asset stages may resume automatically.
                # Started Agent episodes and semantic rejections are not resampled.
                for path in sorted((a.run/'worlds').glob('*/result.json')):
                    result=read_json(path)
                    if 'incomplete_generation:assets/' in result.get('error','') and not result.get('tasks'):
                        archived=path.with_name('result.before_completion_repair.json')
                        if archived.exists():
                            continue
                        path.rename(archived)
                gate_path=a.run/'pilot_gate.json'
                if gate_path.exists() and not read_json(gate_path)['passed']:
                    archived=gate_path.with_name('pilot_gate.before_completion_repair.json')
                    if not archived.exists():
                        gate_path.rename(archived)
            try:
                result=asyncio.run(Pilot(a.run).run_all(a.phase))
            finally:
                report(a.run)
    print(canonical(result),flush=True)


if __name__=='__main__':
    main()
