#!/usr/bin/env python3
"""Offline AST-equivalent correction of AWM's multiline engine wrapper issue."""
import argparse
import asyncio
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from awm_new_pilot import DEFAULT_RUN, freeze, load_manifest, normalize_engine_layout, task_spec_candidate
from harness_awm.contracts import RunConfig
from harness_awm.environment import AWMEnvAdapter
from harness_awm.io import file_hash,read_json


async def repair(run,scenario):
    run=Path(run).resolve()
    m=load_manifest(run,False)
    folder=Path(m['data'])/'worlds'/scenario
    old=read_json(folder/'environment.r0.json')
    fixed=normalize_engine_layout(old['full_code'])
    if fixed==old['full_code']:
        raise ValueError('no_multiline_engine_repair_needed')
    env={**old,'full_code':fixed}
    freeze(folder/'environment.r0.adapter_v1.json',env)
    task=task_spec_candidate(m,folder,read_json(folder/'tasks.json')['tasks'][0],env,'0.adapter_v1')
    adapter=AWMEnvAdapter(task,RunConfig.model_validate(m['config']),folder/'smoke.r0.adapter_v1')
    try:
        await adapter.prepare()
        if not adapter.tools:
            raise ValueError('empty_catalog')
        freeze(folder/'tool_catalog.json',adapter.tools)
    finally:
        await adapter.close()
    freeze(folder/'environment.json',env)
    paths=[folder/x for x in ['tasks.json','schema.json','sample.json','spec.json','initial.db','environment.json','tool_catalog.json']]
    paths+=sorted((folder/'verifiers').glob('*.json'))
    init=read_json(folder/'initialize.r0/output.json')
    freeze(folder/'adapter_repair.json',dict(source_sha256=file_hash(folder/'environment.r0.json'),
        repaired_sha256=file_hash(folder/'environment.json'),ast_equivalent=True,paid_requests=0,
        verification='actual isolated MCP startup and catalog',tools=len(adapter.tools)))
    freeze(folder/'ready.json',dict(hashes={str(p.relative_to(folder)):file_hash(p) for p in paths},
        initialization=init,startup=True,environment_acceptance='pending_real_write_reset_controls',
        adapter_repair_ref=str(folder/'adapter_repair.json')))
    print({'scenario':scenario,'tools':len(adapter.tools),'paid_requests':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--scenario',required=True)
    a=parser.parse_args()
    asyncio.run(repair(a.run,a.scenario))
