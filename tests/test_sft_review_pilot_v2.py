import copy
import contextlib
import io
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'sft_review_v2', Path(__file__).resolve().parents[1] / 'scripts/sft_review_pilot.py'
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

RULE_IDS = [
    'R1_task_scope', 'R2_grounding', 'R3_protocol', 'R4_side_effects',
    'R5_answer_fidelity', 'R6_integrity',
]


class ReviewPilotV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out_patch = patch.object(m, 'OUT', self.root)
        self.out_patch.start()
        self.addCleanup(self.out_patch.stop)
        self.folder = self.root / 'materials' / 't0001'
        m.write(self.folder / 'pack.json', {
            'messages': {}, 'turns': [
                {'turn_id': 1, 'output': 'invalid call'},
                {'turn_id': 2, 'output': 'corrected call'},
                {'turn_id': 3, 'output': 'answer'},
            ],
        })
        m.write(self.folder / 'checks.json', {'errors': []})
        m.write(self.folder / 'rules.json', {
            'schema_version': 'sft-review-v2', 'rule_ids': RULE_IDS,
        })
        self.ref = str(self.folder / 'pack.json') + '#/turns/0/output'
        self.issue = {
            'tag': 'invalid_argument', 'turn_ids': [1], 'evidence_refs': [self.ref],
            'reason': 'The malformed call was corrected without side effects.',
            'rule_id': 'R3_protocol', 'certainty': 'confirmed', 'severity': 'minor',
        }
        self.record = {
            'id': 't0001', 'decision': 'keep', 'reason': 'The error was recovered.',
            'coverage_turn_ids': [1, 2, 3], 'evidence_refs': [self.ref],
            'exclude_loss_turn_ids': [1], 'issues': [self.issue],
        }

    def validate(self, record):
        return m.validate(record, 't0001', self.folder)

    def assert_invalid(self, record):
        with self.assertRaises((ValueError, KeyError, IndexError, TypeError)):
            self.validate(record)

    def test_recovered_confirmed_minor_can_mask_only_bad_output(self):
        self.validate(self.record)
        # The default folder path must use the selected run's OUT as well.
        m.validate(self.record, 't0001')

    def test_uncertain_issue_cannot_support_exclusion(self):
        bad = copy.deepcopy(self.record)
        bad['issues'][0]['certainty'] = 'uncertain'
        self.assert_invalid(bad)
        bad['exclude_loss_turn_ids'] = []
        self.validate(bad)

    def test_exclusion_requires_confirmed_issue_for_each_masked_turn(self):
        bad = copy.deepcopy(self.record)
        second = copy.deepcopy(self.issue)
        second.update(turn_ids=[2], certainty='uncertain')
        bad['issues'].append(second)
        bad['exclude_loss_turn_ids'] = [1, 2]
        self.assert_invalid(bad)

    def test_minor_alone_cannot_reject_and_major_cannot_keep(self):
        bad = copy.deepcopy(self.record)
        bad['decision'] = 'reject'
        self.assert_invalid(bad)
        bad['issues'][0]['severity'] = 'major'
        self.validate(bad)
        bad['decision'] = 'keep'
        self.assert_invalid(bad)

    def test_keep_cannot_mask_every_output(self):
        bad = copy.deepcopy(self.record)
        bad['issues'][0]['turn_ids'] = [1, 2, 3]
        bad['exclude_loss_turn_ids'] = [1, 2, 3]
        self.assert_invalid(bad)

    def test_issue_fields_are_required_and_values_are_restricted(self):
        for field, invalid in [('rule_id', 'R99_unknown'),
                               ('certainty', 'likely'), ('severity', 'critical')]:
            with self.subTest(field=field, case='invalid'):
                bad = copy.deepcopy(self.record)
                bad['issues'][0][field] = invalid
                self.assert_invalid(bad)
            with self.subTest(field=field, case='missing'):
                bad = copy.deepcopy(self.record)
                del bad['issues'][0][field]
                self.assert_invalid(bad)

    def test_rules_reference_must_point_to_existing_field(self):
        record = copy.deepcopy(self.record)
        ref = str(self.folder / 'rules.json') + '#/rule_ids/2'
        record['evidence_refs'] = [ref]
        record['issues'][0]['evidence_refs'] = [ref]
        self.assert_invalid(record)
        record['evidence_refs'].append(self.ref)
        record['issues'][0]['evidence_refs'].append(self.ref)
        self.validate(record)
        record['issues'][0]['evidence_refs'] = [str(self.folder / 'rules.json') + '#/rule_ids/99']
        self.assert_invalid(record)

    def test_v1_without_rules_retains_old_issue_format_and_decisions(self):
        (self.folder / 'rules.json').unlink()
        record = copy.deepcopy(self.record)
        for field in ('rule_id', 'certainty', 'severity'):
            del record['issues'][0][field]
        self.validate(record)
        record['decision'] = 'reject'
        self.validate(record)

    def test_retrieval_scope_does_not_require_listing_every_result(self):
        # The task requests searching all records, while the final answer reports
        # a relevant subset; no independent issue is asserted by this record.
        record = copy.deepcopy(self.record)
        record.update(reason='The requested search was completed.', issues=[],
                      exclude_loss_turn_ids=[])
        self.validate(record)

    def test_evidence_audit_overrides_primary_without_inflating_blind_disagreements(self):
        pack = m.read(self.folder / 'pack.json')
        pack['task'] = 'Complete the requested task.'
        for turn in pack['turns']:
            turn['input_refs'] = []
        m.write(self.folder / 'pack.json', pack)
        m.write(self.root / 'reviews/primary/t0001.json', self.record)
        corrected = copy.deepcopy(self.record)
        corrected.update(decision='reject', adjudication_reason='Evidence shows an unresolved issue.',
                         adjudication_source='spot_check')
        corrected['issues'][0]['severity'] = 'major'
        m.write(self.root / 'reviews/adjudication/t0001.json', corrected)
        manifest = {'selected': [{'id': 't0001', 'stratum': 'ordinary', 'task_id': 'task',
                                  'scenario': 'scenario', 'turn_count': 3}],
                    'second_review_ids': [], 'created_at': m.now(),
                    'config': {'model': 'test-model'}}
        state = {'primary': {'done': ['t0001'], 'missing': [], 'invalid': {}},
                 'secondary': {'done': [], 'missing': [], 'invalid': {}}}
        with patch.object(m, 'verified_manifest', return_value=manifest), \
             patch.object(m, 'status', return_value=state), \
             patch.object(m, 'verify_sources', return_value={'files_checked': 0, 'changed': []}), \
             contextlib.redirect_stdout(io.StringIO()):
            m.finalize()
            self.assertEqual(m.read(self.root / 'reject.json'), ['t0001'])
            summary = m.read(self.root / 'summary.json')
            self.assertEqual(summary['disagreements'], 0)
            self.assertEqual(summary['spot_check_adjudications'], 1)
            self.assertEqual(m.read(self.root / 'human_review.json')[0]['human_decision'], '')
            del corrected['adjudication_reason']
            m.write(self.root / 'reviews/adjudication/t0001.json', corrected)
            with self.assertRaisesRegex(ValueError, 'adjudication must explain'):
                m.finalize()


if __name__ == '__main__':
    unittest.main()
