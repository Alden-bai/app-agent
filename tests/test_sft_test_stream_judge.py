import sqlite3
from pathlib import Path

from harness_awm.io import write_json
from scripts import sft_test_stream_judge_20261110 as stream


def test_discovery_waits_for_done_and_archive_and_deduplicates(tmp_path, monkeypatch):
    (tmp_path / 'test').mkdir()
    episode = tmp_path / 'archive'
    episode.mkdir()
    source = tmp_path / 'episode'
    source.symlink_to(episode, target_is_directory=True)
    write_json(episode / 'episode_result.json', {'execution_integrity': 'valid', 'terminal_state_status': 'trusted'})
    write_json(episode / 'evidence_seal.json', {})
    monkeypatch.setattr(stream.r, 'evaluation_score', lambda result: None)
    item = {'task_instance_id': 'dataset:scenario:1', 'candidate': 'best', 'path': str(source)}
    with sqlite3.connect(tmp_path / 'test/queue.sqlite') as db:
        db.execute('CREATE TABLE jobs(task TEXT,state TEXT)')
        db.execute("INSERT INTO jobs VALUES(?,'claimed')", (item['task_instance_id'],))
    assert stream.discover(tmp_path, [item], set())[0] == []
    with sqlite3.connect(tmp_path / 'test/queue.sqlite') as db:
        db.execute("UPDATE jobs SET state='done'")
    assert stream.discover(tmp_path, [item], set())[0] == []
    write_json(episode / 'storage_archive.json', {})
    rows, counts, skipped = stream.discover(tmp_path, [item], set())
    assert len(rows) == 1 and counts == {'done': 1} and not skipped
    assert rows[0]['task_id'] == 'scenario:1'
    assert stream.discover(tmp_path, [item], {rows[0]['key']})[0] == []
    write_json(episode / 'episode_result.json', {'execution_integrity': 'invalid', 'terminal_state_status': 'untrusted'})
    rows, _, skipped = stream.discover(tmp_path, [item], set())
    assert not rows and skipped[item['task_instance_id']] == 'untrusted_execution'


def test_shared_gate_enforces_tpm_and_counts_unknown_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(stream.j.previous, 'RUN', tmp_path)
    write_json(tmp_path / 'transport_settings.json', {'rpm': 5000, 'workers': 16})
    gate = stream.j.previous.SmoothGate(tmp_path / 'rate.sqlite')
    with sqlite3.connect(gate.path) as db:
        db.execute('UPDATE pacing SET tpm=2000000 WHERE id=1')
        db.execute("INSERT INTO events(started,tokens,status) VALUES(100,1990000,'unknown_reserved')")
    assert gate.delay(110, 10001) > 0
    assert gate.delay(110, 10000) == 0
    assert gate.delay(161, 95000) == 0
    with sqlite3.connect(gate.path) as db:
        db.execute("INSERT INTO events(started,tokens,status) VALUES(161,90000,'reserved')")
        event = db.execute('SELECT max(id) FROM events').fetchone()[0]
    gate.finish(event, 100)
    with sqlite3.connect(gate.path) as db:
        assert db.execute('SELECT tokens FROM events WHERE id=?', (event,)).fetchone()[0] == 90000


def test_shared_gate_enforces_5000_rpm(tmp_path, monkeypatch):
    monkeypatch.setattr(stream.j.previous, 'RUN', tmp_path)
    write_json(tmp_path / 'transport_settings.json', {'rpm': 5000, 'workers': 16})
    gate = stream.j.previous.SmoothGate(tmp_path / 'rate.sqlite')
    with sqlite3.connect(gate.path) as db:
        db.execute('UPDATE pacing SET tpm=2000000 WHERE id=1')
        db.executemany("INSERT INTO events(started,tokens,status) VALUES(100,1,'reserved')", [()] * 5000)
    assert gate.delay(110, 1) > 0
    assert gate.delay(161, 1) == 0
