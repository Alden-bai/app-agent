"""Invoke every exposed tool's basic parameter contract on an audit instance.

Missing-required-field calls and empty legal requests test the actual MCP/HTTP
boundary. Business semantics require additional task-specific checks. Every
request has empty arguments, so no hidden database parameter can enter it.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

from awm_audit import runtime as legacy
from awm_audit.common import backup, digest, now, read_json, save_json
from .common import business_state


async def basic_contracts(run, scenario, initial, folder):
    import httpx
    import jsonschema
    folder = Path(folder).resolve()
    result_file = folder / 'result.json'
    if result_file.exists():
        return read_json(result_file)
    folder.mkdir(parents=True, exist_ok=True)
    work = folder / 'working.db'
    backup(initial, work)
    baseline = digest(business_state(work))
    port = legacy.free_port()
    log = (folder / 'server.log').open('w')
    process = None
    records = []
    failure = None
    try:
        script = legacy.server_script(run, scenario, work, folder, port)
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'LANG', 'LC_ALL', 'TZ')}
        env.update(HOST='127.0.0.1', PORT=str(port), DATABASE_PATH=f'sqlite:///{work}')
        process = subprocess.Popen([sys.executable, script], cwd=folder, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        async with httpx.AsyncClient(timeout=1, trust_env=False) as http:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f'Contract MCP server exited: {process.returncode}')
                try:
                    if (await http.get(f'http://127.0.0.1:{port}/openapi.json')).status_code == 200:
                        break
                except (httpx.HTTPError, OSError):
                    pass
                await asyncio.sleep(0.25)
            else:
                raise TimeoutError('Contract MCP startup exceeded 30 seconds')
        async with legacy.session(f'http://127.0.0.1:{port}/mcp') as client:
            tools = (await client.list_tools()).tools
            save_json(folder / 'tools.json', [t.model_dump(mode='json') for t in tools])
            for index, tool in enumerate(tools):
                case_folder = folder / f'tool_{index:04d}'
                case_folder.mkdir(exist_ok=True)
                before = digest(business_state(work))
                # A permitted no-argument write can change the audit state.
                # Subsequent probes must begin from the same clean business state.
                if before != baseline:
                    backup(initial, work)
                    before = digest(business_state(work))
                    if before != baseline:
                        raise RuntimeError('Could not reset tool-contract state')
                backup(work, case_folder / 'initial.db')
                schema = tool.inputSchema
                jsonschema.Draft202012Validator.check_schema(schema)
                valid_input = jsonschema.Draft202012Validator(schema).is_valid({})
                started = time.monotonic()
                response = await client.call_tool(tool.name, {})
                raw = response.model_dump(mode='json')
                save_json(case_folder / 'request.json', {'tool': tool.name, 'arguments': {},
                          'source': 'public inputSchema empty-object contract probe'})
                save_json(case_folder / 'response.json', raw)
                backup(work, case_folder / 'final.db')
                after = digest(business_state(work))
                checks = {'input_empty_object_schema_valid': valid_input,
                          'invalid_input_rejected': response.isError if not valid_input else None,
                          'invalid_input_has_no_business_writes': before == after if not valid_input else None}
                hard = not valid_input and (not response.isError or before != after)
                record = {'tool': tool.name, 'at': now(), 'schema_sha256': digest(schema),
                          'status': 'contract_defect' if hard else ('basic_contract_checked' if not valid_input or not response.isError else 'legal_empty_input_error_needs_review'),
                          'checks': checks, 'is_error': response.isError,
                          'business_state_before': before, 'business_state_after': after,
                          'evidence': str(case_folder), 'elapsed_seconds': time.monotonic() - started,
                          'semantic_status': 'not_checked',
                          'note': 'Input/error/persistence boundary only; no claim of complete query/write business semantics.'}
                records.append(record)
                save_json(folder / 'progress.json', records)
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'error': str(exc)}
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        log.close()
    result = {'scenario': scenario, 'status': 'checked' if failure is None else 'incomplete',
              'all_exposed_tools_invoked': failure is None,
              'all_basic_contracts_passed': failure is None and all(r['status'] == 'basic_contract_checked' for r in records),
              'tools': records, 'failure': failure,
              'scope': 'All discovered tools; empty input validity, actual response, and invalid-input business side effects.',
              'remaining': ['task dependency and alternative-path query/write/error business semantics']}
    save_json(result_file, result)
    return result
