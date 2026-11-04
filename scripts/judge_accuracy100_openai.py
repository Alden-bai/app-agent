#!/usr/bin/env python3
"""OpenAI rerun of the same frozen100; preserve DeepSeek records and cumulative cap."""
import argparse
import asyncio
import copy
import fcntl
import json
import os
import shutil
from pathlib import Path

import judge_accuracy100 as base
from harness_awm.backend import Budget, api_request
from harness_awm.contracts import APIConfig, RunConfig
from harness_awm.io import canonical, file_hash, now, read_json, write_json

RUN = base.ROOT / 'runs/openai_gpt6_luna_judge_accuracy100_20261103'
PRIOR = base.ROOT / 'runs/deepseek_judge_accuracy100_20261103'


def credentials():
    from dotenv import dotenv_values
    values = dotenv_values(base.ROOT / '.env', interpolate=False)
    if not values.get('OPENAI_API_KEY') or values.get('OPENAI_MODEL') != 'gpt-6-luna':
        raise ValueError('OpenAI_key_or_model_not_configured')
    os.environ['AWM_OPENAI_STUDY_KEY'] = values['OPENAI_API_KEY']
    return APIConfig(base_url=values['OPENAI_BASE_URL'], model=values['OPENAI_MODEL'],
        api_key_env='AWM_OPENAI_STUDY_KEY', reasoning_effort='none', timeout=180, context_limit=128000)


def prepare(run):
    run = Path(run).resolve()
    if (run / 'manifest.json').exists():
        assert file_hash(run / 'manifest.json') == read_json(run / 'manifest.sha256.json')['sha256']
        return read_json(run / 'manifest.json')
    if not read_json(run / 'connectivity/result.json')['valid']:
        raise ValueError('connectivity_not_validated')
    old = read_json(PRIOR / 'manifest.json')
    assert file_hash(PRIOR / 'manifest.json') == read_json(PRIOR / 'manifest.sha256.json')['sha256']
    api = credentials()
    config = RunConfig.model_validate(old['config']).model_copy(update={'actor_api': api, 'judge_api': api})
    m = copy.deepcopy(old)
    for rel, expected in old['input_hashes'].items():
        source = PRIOR / rel
        assert file_hash(source) == expected
        target = run / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for task in m['tasks']: task['input'] = str(run / 'inputs' / task['id'])
    m.update(created_at=now(), config=config.model_dump(), prior_manifest_sha256=file_hash(PRIOR / 'manifest.json'),
        provider='OpenAI', model='gpt-6-luna', reasoning_effort='none',
        reference_protocol=old['reference_protocol'].replace('DeepSeek', 'GPT-6 Luna'),
        transport_policy='Chat Completions; original official prompt and temperature retained; explicit reasoning_effort=none; max_completion_tokens unchanged; no vendor thinking body; no automatic retries')
    paths = [*Path(base.ROOT / 'harness_awm').glob('*.py'), Path(base.__file__), Path(__file__),
             base.ROOT / 'agent-world-model/awm/core/verify.py']
    m['implementation_hashes'] = {str(p): file_hash(p) for p in paths}
    write_json(run / 'manifest.json', m)
    write_json(run / 'manifest.sha256.json', {'sha256': file_hash(run / 'manifest.json')})
    return m


class OpenAIStudy(base.Study):
    def __init__(self, run):
        self.run = Path(run).resolve(); self.m = prepare(self.run)
        self.config = RunConfig.model_validate(self.m['config'])
        assert credentials() == self.config.actor_api
        for rel, expected in self.m['input_hashes'].items():
            assert file_hash(self.run / rel) == expected
        for path, expected in self.m['implementation_hashes'].items():
            assert file_hash(path) == expected, path
        self.budget = Budget(self.run / 'budget.sqlite', self.config.max_physical_requests, base.CAP)
        self.api_sem = asyncio.Semaphore(4); self.judge_sem = asyncio.Semaphore(2); self.env_sem = asyncio.Semaphore(4)

    async def llm(self, folder, stage, messages, max_output=4096):
        folder = Path(folder)
        if (folder / 'parsed.json').exists(): return read_json(folder / 'parsed.json')
        if (folder / 'started.json').exists(): raise ValueError('attempt_already_started_no_retry')
        write_json(folder / 'started.json', {'started_at': now(), 'stage': stage})
        request = {'model': self.config.actor_api.model, 'messages': messages,
                   'temperature': 0, 'max_completion_tokens': max_output,
                   'response_format': {'type': 'json_object'}}
        async with self.api_sem:
            response = await api_request(self.config.actor_api, request, self.budget,
                str(folder.relative_to(self.run)), stage, folder)
        choice = response['choices'][0]
        if choice['finish_reason'] != 'stop': raise ValueError('incomplete_reference_output')
        value = json.loads(choice['message']['content'])
        if not isinstance(value, dict): raise ValueError('reference_not_object')
        write_json(folder / 'parsed.json', value)
        return value

    async def reference(self, *args):
        value = await super().reference(*args)
        value['reference_type'] = 'rubric_and_evidence_based_two_pass_gpt6_luna_reference_not_human_gold'
        return value

    def report(self):
        report = super().report()
        report['model'] = 'gpt-6-luna'; report['reasoning_effort'] = 'none'
        report['limitations'] = [x.replace('DeepSeek', 'GPT-6 Luna') for x in report['limitations']]
        report['historical_budget_occupied'] = 387453
        report['current_run_known_tokens'] = sum(v.get('total_tokens', 0) for k,v in report['by_stage'].items() if not k.startswith('prior_'))
        report['current_run_budget_occupied'] = report['budget']['tokens_accounted_or_reserved'] - 387453
        write_json(self.run / 'summary.json', report)
        p = self.run / 'report_zh.md'
        p.write_text(p.read_text().replace('DeepSeek Flash', 'GPT-6 Luna').replace('DeepSeek', 'GPT-6 Luna'))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    parser.add_argument('--run', type=Path, default=RUN)
    args = parser.parse_args()
    if args.action == 'prepare':
        m = prepare(args.run); print(canonical({'tasks': len(m['tasks']), 'model': m['model']})); return
    study = OpenAIStudy(args.run)
    with (args.run / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = asyncio.run(study.run_all()) if args.action == 'run' else study.report()
    print(canonical(result), flush=True)


if __name__ == '__main__': main()
