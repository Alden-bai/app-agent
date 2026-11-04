#!/usr/bin/env python3
"""Offline discovery audit of frozen runtime catalogs; never invokes models or business tools."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness_awm.io import Recorder, canonical, file_hash, read_json, write_json
from harness_awm.protocol_v2 import ObservationsV2, ToolCatalog


def audit(source, output):
    cases = []
    for path in sorted(Path(source).glob('tasks/*/episode/tool_catalog.json')):
        tools = read_json(path)['tools']
        catalog = ToolCatalog(tools, 8000)
        names, cursor = [], None
        while True:
            page = catalog.page(cursor=cursor, names_only=True)
            names.extend(t['name'] for t in page['tools'])
            cursor = page['next_cursor']
            if cursor is None: break
        if names != sorted(t['name'] for t in tools):
            raise ValueError('incomplete_catalog:' + str(path))
        with tempfile.TemporaryDirectory(prefix='awm-catalog-audit-') as tmp:
            obs = ObservationsV2(Recorder(tmp, path.parents[1].name), 8000, 4000)
            for tool in tools:
                name = tool['name']
                if catalog.page(name)['tools'][0]['name'] != name:
                    raise ValueError('exact_search_not_first:' + name)
                visible, _ = obs.present(catalog.schema(name))
                if visible['truncated']:
                    text, offset = '', 0
                    while True:
                        page = obs.read({'observation_id':visible['observation_id'], 'offset':offset, 'limit':999999})
                        text += page['content']
                        if not page['has_more']: break
                        offset = page['next_offset']
                    actual = json.loads(text)
                else:
                    actual = visible['content']
                if actual != tool:
                    raise ValueError('schema_changed:' + name)
        cases.append({'id':path.parents[1].name,'catalog_sha256':file_hash(path),
                      'tools':len(tools),'catalog_chars':len(canonical({'tools':tools})),
                      'all_names_reachable':True,'all_exact_searches_passed':True,'all_schemas_lossless':True})
    result = {'source':str(Path(source).resolve()),'model_requests':0,'business_calls':0,
              'catalogs':len(cases),'tools_verified':sum(c['tools'] for c in cases),'cases':cases}
    write_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/openai_gpt6_luna_judge_accuracy100_20261103')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.source,args.output)
    print(canonical({k:v for k,v in result.items() if k!='cases'}))
