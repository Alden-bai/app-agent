import copy
import json
import sqlite3
import asyncio
import errno

import pytest

from harness_awm.io import canonical
from scripts.sft_remaining_recheck_20261110 import ExpandedEvidenceStore, expanded_view
from scripts.sft_remaining_recheck_20261110 import FullOutputTransport
from harness_awm.contracts import APIConfig
from scripts.sft_remaining_recheck_20261110 import sealed_copy
from harness_awm.io import seal, verify_seal, write_json, file_hash


@pytest.fixture
def store(tmp_path):
    paths = [tmp_path / (x + '.db') for x in ('initial', 'final')]
    for path in paths:
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE items(id INTEGER PRIMARY KEY)')
    return ExpandedEvidenceStore(*paths)


def test_expanded_pages_reassemble_exact_unicode_source(store):
    original = store.add('tool_result', {'text': '中文abc' * 6000})
    before = copy.deepcopy(original)
    offset, chunks = 0, []
    while offset is not None:
        page, truncated = store.read_evidence(original['id'], offset)
        assert len(page['text'].encode()) <= 8000
        assert page['next_offset'] is None or page['next_offset'] > offset
        chunks.append(page['text']);offset = page['next_offset']
    assert len(chunks[0].encode()) > 2500
    assert ''.join(chunks) == canonical(original['content'])
    assert original == before


def test_failed_evidence_cannot_be_promoted(store):
    record = store.add('query_error', None, valid=False, error='bad SQL')
    with pytest.raises(ValueError, match='cannot_promote_failed_evidence'):
        store.read_evidence(record['id'])


def test_larger_page_visible_and_provenance_preserved(store):
    runtime = store.add('runtime', {'timezone': 'Asia/Shanghai'})
    source = store.add('query', {'rows': [['x' * 12000]]},
                       tool='query_snapshot', arguments={'snapshot': 'initial', 'sql': 'SELECT 1'})
    page = store.call('read_evidence', {'evidence_id': source['id']}, objective='read', affected_items=['R01'])
    before = copy.deepcopy(store.records)
    view = expanded_view(store.records)
    assert view[runtime['id']]['content'] == runtime['content']
    assert not view[page['id']]['view_truncated']
    assert view[page['id']]['content'] == page['content']
    assert view[source['id']]['arguments']['snapshot'] == 'initial'
    assert view[page['id']]['next_offset'] == page['content']['next_offset']
    assert before == store.records


def test_context_pressure_keeps_runtime_and_page_cursors(store):
    clock = store.add('runtime', {'actual_start_utc': '2026-11-09T00:00:00Z'})
    for _ in range(40):
        source = store.add('query', {'rows': [['x' * 12000]]})
        page = store.call('read_evidence', {'evidence_id': source['id']}, objective='read', affected_items=['R01'])
    view = expanded_view(store.records)
    assert view[clock['id']]['content'] == clock['content']
    assert len(view) == len(store.records)
    assert view[page['id']]['source_evidence_id'] == source['id']
    assert any(row['view_truncated'] for row in view.values())


def test_context_preflight_counts_tokens_not_utf8_bytes(tmp_path):
    transport = FullOutputTransport(None, None, tmp_path)
    captured = []
    async def exact(config, wire, *args, **kw):
        captured.append((config, wire))
        return {'ok': True}
    transport.exact = exact
    config = APIConfig(context_limit=128000)
    wire = {'messages': [{'role': 'user', 'content': 'a ' * 65000}]}
    result = asyncio.run(transport.request(config, wire, None, 'k', 'judge', tmp_path/'call'))
    assert result == {'ok': True}
    assert captured[0][0].context_limit is None
    assert config.context_limit == 128000
    assert captured[0][1]['messages'] == wire['messages']
    check = json.loads((tmp_path/'call/context_check.json').read_text())
    assert check['byte_estimate'] > 128000
    assert check['tokenizer_reservation'] < 95000


def test_truly_oversized_request_never_reaches_api(tmp_path):
    transport = FullOutputTransport(None, None, tmp_path)
    async def forbidden(*args, **kw):
        raise AssertionError('API must not be called')
    transport.exact = forbidden
    with pytest.raises(ValueError, match='tokenized_request_exceeds_safe_context'):
        asyncio.run(transport.request(APIConfig(context_limit=128000),
            {'messages': [{'role':'user','content':'a '*150000}]},None,'k','judge',tmp_path/'call'))


def test_oversized_evidence_preview_shrinks_without_changing_source_metadata(tmp_path):
    transport = FullOutputTransport(None, None, tmp_path)
    captured = []
    async def exact(config, wire, *args, **kw):
        captured.append(wire);return {'ok': True}
    transport.exact = exact
    payload = {'evidence': {'E1': {'id':'E1','kind':'query','content':'a '*150000,
               'view_truncated':False,'arguments':{'snapshot':'final'}}}, 'rubric':{'task_id':'test'}}
    wire = {'messages':[{'role':'user','content':canonical(payload)}]}
    before = copy.deepcopy(wire)
    asyncio.run(transport.request(APIConfig(context_limit=128000),wire,None,'k','judge',tmp_path/'call'))
    sent = json.loads(captured[0]['messages'][0]['content'])
    assert sent['evidence']['E1']['view_truncated']
    assert sent['evidence']['E1']['arguments'] == {'snapshot':'final'}
    assert sent['rubric'] == payload['rubric']
    assert wire == before


def test_cross_device_copy_preserves_seal_and_result_isolation(tmp_path, monkeypatch):
    source = tmp_path/'source';source.mkdir()
    (source/'initial.db').write_bytes(b'initial bytes')
    (source/'final.db').write_bytes(b'final bytes')
    write_json(source/'episode_result.json', {'evaluation_status':'pending'})
    original_seal = seal(source)
    original_result = file_hash(source/'episode_result.json')
    def cross_device(*args):
        raise OSError(errno.EXDEV, 'Invalid cross-device link')
    monkeypatch.setattr('scripts.sft_remaining_recheck_20261110.os.link', cross_device)
    target = tmp_path/'target'
    provenance = sealed_copy(source, target)
    assert verify_seal(target) == original_seal == verify_seal(source)
    assert set(provenance['cross_device_copies']) == {'initial.db','final.db'}
    write_json(target/'episode_result.json', {'evaluation_status':'valid'})
    assert file_hash(source/'episode_result.json') == original_result
