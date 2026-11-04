#!/usr/bin/env python3
"""Reproduce failed GETs in an isolated ASGI client to retain Python tracebacks."""
import json
from pathlib import Path
import runpy
import sys
import traceback

from fastapi.testclient import TestClient

folder = Path(sys.argv[1]).resolve()
result = json.loads((folder / 'result.json').read_text())
namespace = runpy.run_path(str(folder / 'runtime/server.py'), run_name='retest_diagnostic')
if '--routes' in sys.argv:
    from starlette.routing import Match
    matches = []
    for call in result['calls']:
        scope = dict(type='http', method='GET', path=call['path'], root_path='')
        route = next((r for r in namespace['app'].routes if r.matches(scope)[0] == Match.FULL), None)
        if route is not None and route.path != call['path']:
            matches.append(dict(tool=call['tool'], intended_path=call['path'], actual_path=route.path,
                actual_handler=route.endpoint.__name__, returned_mcp_error=call['response'].get('isError')))
    (folder / 'route_matches.json').write_text(json.dumps(matches, ensure_ascii=False, indent=2) + '\n')
    raise SystemExit(0)
diagnostics = []
with TestClient(namespace['app'], raise_server_exceptions=True) as client:
    for call in result['calls']:
        if not call['response'].get('isError'):
            continue
        item = dict(tool=call['tool'], path=call['path'])
        try:
            response = client.get(call['path'])
            item.update(status=response.status_code, body=response.text)
        except Exception as exc:
            item.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        diagnostics.append(item)
(folder / 'diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + '\n')
