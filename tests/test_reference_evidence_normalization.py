import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from finish_openai_judge_accuracy100 import normalize_evidence


class NormalizationTests(unittest.TestCase):
    def test_only_explicit_sources_are_normalized_and_labels_preserved(self):
        candidate = {'task_label': 'fail', 'items': [
            {'id': 'a', 'status': 'fail', 'reason': 'q_final and trajectory show missing rows'},
            {'id': 'b', 'status': 'pass', 'reason': 'source definition',
             'evidence_ids': ['contract:sample', 'schema:items']}]}
        before = copy.deepcopy(candidate)
        evidence = {k: {'valid': True} for k in ['q_final', 'trajectory', 'contract', 'schema']}
        result, changes = normalize_evidence(candidate, evidence, 'sample', {'items'})
        self.assertEqual(candidate, before)
        self.assertEqual(result['task_label'], 'fail')
        self.assertEqual(result['items'][0]['status'], 'fail')
        self.assertEqual(result['items'][0]['evidence_ids'], ['q_final', 'trajectory'])
        self.assertEqual(result['items'][1]['evidence_ids'], ['contract', 'schema'])
        self.assertTrue(changes)

    def test_unsupported_sources_remain_invalid(self):
        for item in [
            {'reason': 'q_final_extra proves it'},
            {'reason': 'source', 'evidence_ids': ['contract:wrong']},
            {'reason': 'source', 'evidence_ids': ['schema:absent']},
        ]:
            with self.assertRaises(ValueError):
                normalize_evidence({'items': [{'id': 'a', **item}]},
                                   {'q_final': {'valid': True}, 'contract': {}, 'schema': {}},
                                   'sample', {'items'})
