import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sft_full', Path(__file__).resolve().parents[1] / 'scripts/sft_review_full.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FullAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'full'; self.base = self.root / 'pilot'
        self.out.mkdir(); self.base.mkdir()
        for name, value in [('OUT', self.out), ('BASE', self.base)]:
            x = patch.object(m, name, value); x.start(); self.addCleanup(x.stop)
        self.folder = self.out / 'materials/t0001'
        m.write(self.folder / 'pack.json', {'task': 'task', 'messages': {},
              'turns': [{'turn_id': 1, 'output': 'bad', 'input_refs': []},
                        {'turn_id': 2, 'output': 'good', 'input_refs': []}]})
        m.write(self.folder / 'checks.json', {'errors': []})
        m.write(self.folder / 'rules.json', {'schema_version': 'sft-review-v2',
                'rule_revision': m.REVISION, 'rule_ids': ['R3_protocol', 'R6_integrity']})
        self.ref = str(self.folder / 'pack.json') + '#/turns/0/output'
        self.record = {'id': 't0001', 'decision': 'keep', 'reason': 'recovered',
                       'evidence_refs': [self.ref], 'coverage_turn_ids': [1, 2],
                       'exclude_loss_turn_ids': [1], 'issues': [
                           {'rule_id': 'R3_protocol', 'certainty': 'confirmed', 'severity': 'minor',
                            'tag': 'invalid_json', 'turn_ids': [1], 'reason': 'recovered', 'evidence_refs': [self.ref]}]}
        self.frozen = {'config': {'model': 'gpt-6.1-terra', 'max_attempts': 3},
                       'materials': {'t0001': m.file_hashes(self.folder)},
                       'frozen_files': {str(self.out / 'prompt.txt'): 'prompt', str(self.out / 'rules.json'): 'rules'},
                       'plans': {'primary': ['t0001'], 'targeted': [], 'secondary': []},
                       'preparation_incomplete': {}, 'selected': [{'id': 't0001', 'episode': 'episode'}], 'reuse_ids': []}

    def test_v21_retains_all_v2_constraints(self):
        m.validate(self.record, 't0001')
        for field, value in [('certainty', 'uncertain'), ('severity', 'major'), ('rule_id', 'unknown')]:
            r = copy.deepcopy(self.record); r['issues'][0][field] = value
            with self.assertRaises(ValueError): m.validate(r, 't0001')

    def test_schema_cannot_silently_downgrade(self):
        r = m.p.read(self.folder / 'rules.json'); r['schema_version'] = 'sft-review-v2.1'
        m.write(self.folder / 'rules.json', r)
        with self.assertRaisesRegex(ValueError, 'strict schema'): m.validate(self.record, 't0001')

    def test_bad_turn_reference_coverage_and_mask_are_detected(self):
        for mutate in [lambda r: r.update(coverage_turn_ids=[1]),
                       lambda r: r.update(coverage_turn_ids=[True, 2]),
                       lambda r: r.update(exclude_loss_turn_ids=[3]),
                       lambda r: r.update(evidence_refs=[self.ref.replace('/0/', '/9/')]),
                       lambda r: r.update(evidence_refs=[self.ref.replace('/0/', '/-1/')]),
                       lambda r: r['issues'][0].update(evidence_refs=[str(self.folder/'pack.json')+'#/task']),
                       lambda r: r['issues'][0].update(certainty='uncertain')]:
            r = copy.deepcopy(self.record); mutate(r)
            with self.assertRaises((ValueError, IndexError, KeyError)): m.validate(r, 't0001')

    def test_reject_category_and_confirmed_vs_uncertain(self):
        r = copy.deepcopy(self.record); r.update(decision='reject', exclude_loss_turn_ids=[])
        r['issues'][0].update(certainty='uncertain', severity='major')
        with self.assertRaisesRegex(ValueError, 'reject category'): m.validate(r, 't0001')
        r['reject_category'] = 'insufficient_evidence'; m.validate(r, 't0001')

    def test_cache_bound_to_material_config_revision_phase(self):
        k = m.key(self.frozen, 't0001', 'primary')
        self.assertNotEqual(k, m.key(self.frozen, 't0001', 'targeted'))
        for field in ('materials', 'config', 'frozen_files'):
            other = copy.deepcopy(self.frozen); other[field]['extra'] = 'changed'
            if field == 'materials': other[field]['t0001']['changed'] = 'hash'
            if field == 'frozen_files': other[field][str(self.out/'prompt.txt')] = 'new'
            self.assertNotEqual(k, m.key(other, 't0001', 'primary'))

    def test_claim_accept_and_resume_prevent_duplicate(self):
        with patch.object(m, 'manifest', return_value=self.frozen):
            job = m.claim('t0001', 'primary', 'agent')
            m.write(Path(job['output']), self.record)
            m.accept('t0001', 'primary')
            self.assertTrue(m.done(self.frozen, 'primary', 't0001'))
            with self.assertRaisesRegex(ValueError, 'already completed'): m.claim('t0001', 'primary')
            self.assertEqual(m.status()['lanes']['primary']['done'], ['t0001'])

    def test_bare_old_json_does_not_become_new_cache(self):
        m.write(m.result_path('primary', 't0001'), self.record)
        self.assertFalse(m.done(self.frozen, 'primary', 't0001'))
        self.assertFalse(m.result_path('primary', 't0001').with_suffix('.receipt.json').exists())

    def test_failed_processing_has_three_attempt_limit_and_never_reject(self):
        with patch.object(m, 'manifest', return_value=self.frozen):
            for _ in range(3):
                m.claim('t0001', 'primary'); m.fail('t0001', 'primary', 'invalid output')
            with self.assertRaisesRegex(ValueError, 'attempt limit'): m.claim('t0001', 'primary')
            s = m.status()['lanes']['primary']
            self.assertEqual(s['exhausted'], ['t0001']); self.assertFalse(s['done'])
            self.assertFalse(m.result_path('primary', 't0001').exists())

    def test_one_item_dispatch_does_not_scan_other_materials(self):
        frozen = copy.deepcopy(self.frozen)
        frozen['frozen_files'] = {}
        frozen['materials']['t0002'] = {'/nonexistent/other_material': 'hash'}
        m.write(self.out/'manifest.json', frozen)
        m.write(self.out/'manifest.sha256.json', {'sha256': m.p.sha(self.out/'manifest.json')})
        m.manifest('t0001')
        with self.assertRaisesRegex(ValueError, 't0002'): m.manifest(full=True)

    def test_targeted_cannot_be_skipped_by_old_primary(self):
        self.frozen['plans']['targeted'] = ['t0001']
        with patch.object(m, 'manifest', return_value=self.frozen):
            j = m.claim('t0001', 'primary'); m.write(j['output'], self.record); m.accept('t0001', 'primary')
            j = m.claim('t0001', 'targeted')
            self.assertEqual(j['lane'], 'targeted')

    def test_impact_scan_reads_final_record_and_detects_any_state_change(self):
        folder = self.base/'materials/t0001'; folder.mkdir(parents=True)
        for name in ('pack.json', 'checks.json', 'rules.json'):
            (folder/name).write_text((self.folder/name).read_text())
        (folder/'private.json').write_text(json.dumps({'initial': {'tables': {'assets': [{'miles': 100}]}},
                                                     'final': {'tables': {'assets': [{'miles': 90}]}}}))
        r = copy.deepcopy(self.record)
        def remap(ref): return ref.replace(str(self.folder), str(folder))
        r['evidence_refs'] = list(map(remap, r['evidence_refs']))
        r['issues'][0]['evidence_refs'] = list(map(remap, r['issues'][0]['evidence_refs']))
        scan = m.impact_scan([{'id': 't0001'}], {'t0001': r})
        self.assertEqual(scan[0]['action'], 'targeted')
        self.assertEqual(scan[0]['changed_tables'], ['assets'])

    def test_writes_outside_run_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'): m.write(self.base/'status.json', {})

    def test_finalize_handles_new_ids_missing_baseline_and_preserves_human(self):
        (self.base/'reviews.jsonl').write_text('')
        self.frozen.update(created_at=m.p.now(), second_review_ids=[])
        with patch.object(m, 'manifest', return_value=self.frozen):
            j = m.claim('t0001', 'primary'); m.write(j['output'], self.record); m.accept('t0001', 'primary')
            m.finalize()
            human = m.p.read(self.out/'human_review.json'); human[0]['human_decision'] = 'keep'; human[0]['human_notes'] = 'manual'
            m.write(self.out/'human_review.json', human)
            (self.out/'human_review.md').write_text('manual markdown')
            m.finalize()
            self.assertEqual(m.p.read(self.out/'human_review.json')[0]['human_notes'], 'manual')
            self.assertEqual((self.out/'human_review.md').read_text(), 'manual markdown')
            self.assertEqual(len((self.out/'reviews.jsonl').read_text().splitlines()), 1)

    def test_adjudication_applies_outside_blind_sample(self):
        (self.base/'reviews.jsonl').write_text('')
        self.frozen.update(created_at=m.p.now(), second_review_ids=[])
        with patch.object(m, 'manifest', return_value=self.frozen):
            j = m.claim('t0001', 'primary'); m.write(j['output'], self.record); m.accept('t0001', 'primary')
            r = copy.deepcopy(self.record)
            r.update(decision='reject', reject_category='confirmed_quality_issue', adjudication_reason='Evidence checked by root')
            r['issues'][0]['severity'] = 'major'
            j = m.claim('t0001', 'adjudication', 'root'); m.write(j['output'], r); m.accept('t0001', 'adjudication')
            m.finalize()
            self.assertEqual(json.loads((self.out/'reviews.jsonl').read_text())['decision'], 'reject')


if __name__ == '__main__': unittest.main()
