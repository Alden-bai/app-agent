import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from harness_awm.backend import Budget, BudgetExceeded
from scripts import sft_optimized_judge_20261110 as opt


class Gate:
    def __init__(self):
        self.reservations = []
        self.backoffs = []

    async def acquire(self, upper):
        self.reservations.append(upper)
        return len(self.reservations)

    def finish(self, *args):
        pass

    def limited(self, *args):
        self.backoffs.append(args)


def response(reason='stop', text='complete assessment'):
    return {'choices': [{'finish_reason': reason, 'message': {'content': text}}],
            'usage': {'total_tokens': 100}}


class OptimizedJudgeTest(unittest.TestCase):
    def test_complete_output_preserves_inputs_and_uses_4096(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = Budget(Path(tmp) / 'budget.sqlite', 32, None)
            messages = [{'role': 'user', 'content': 'all original evidence'}]
            request = {'messages': messages, 'max_completion_tokens': 8192, 'temperature': 1}
            before = copy.deepcopy(request)
            calls = []
            async def api(config, wire, ledger, key, stage, folder, **kw):
                calls.append(wire)
                rid = ledger.reserve(key, stage, wire['messages'], wire['max_completion_tokens'])
                ledger.finish(rid, {'total_tokens': 100})
                return response()
            result = asyncio.run(opt.request_with_fallback(api, Gate(), None, request, budget,
                                                         'key', 'judge', Path(tmp) / 'call'))
            self.assertEqual(result['choices'][0]['finish_reason'], 'stop')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['max_completion_tokens'], 4096)
            self.assertIs(calls[0]['messages'], messages)
            self.assertEqual(request, before)
            self.assertEqual(budget.summary()['physical_requests'], 1)

    def test_truncation_discards_partial_and_accounts_both_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            budget = Budget(job / 'budget.sqlite', 32, None)
            request = {'messages': [{'role': 'user', 'content': 'evidence'}], 'max_completion_tokens': 8192}
            calls = []
            async def api(config, wire, ledger, key, stage, folder, **kw):
                calls.append((wire, key, folder))
                rid = ledger.reserve(key, stage, wire['messages'], wire['max_completion_tokens'])
                ledger.finish(rid, {'total_tokens': 100})
                return response('length', 'PARTIAL MUST NOT BE USED') if len(calls) == 1 else response()
            result = asyncio.run(opt.request_with_fallback(api, Gate(), None, request, budget,
                                                         'key', 'judge', job / 'call'))
            self.assertEqual([c[0]['max_completion_tokens'] for c in calls], [4096, 8192])
            self.assertIs(calls[0][0]['messages'], calls[1][0]['messages'])
            self.assertNotEqual(calls[0][1], calls[1][1])
            self.assertNotEqual(calls[0][2], calls[1][2])
            self.assertEqual(result['choices'][0]['message']['content'], 'complete assessment')
            self.assertEqual(budget.summary()['physical_requests'], 2)
            self.assertEqual(budget.summary()['tokens_accounted_or_reserved'], 200)
            self.assertTrue(opt.r.read_json(job / 'call/output_fallback.json')['partial_output_discarded'])

    def test_fallback_cannot_reset_or_expand_cumulative_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = Budget(Path(tmp) / 'budget.sqlite', 32, None)
            for i in range(31):
                rid = budget.reserve(str(i), 'old_attempt', [], 1)
                budget.finish(rid, {'total_tokens': 1})
            async def api(config, wire, ledger, key, stage, folder, **kw):
                rid = ledger.reserve(key, stage, wire['messages'], wire['max_completion_tokens'])
                ledger.finish(rid, {'total_tokens': 100})
                return response('length')
            request = {'messages': [], 'max_completion_tokens': 8192}
            with self.assertRaises(BudgetExceeded):
                asyncio.run(opt.request_with_fallback(api, Gate(), None, request, budget,
                                                     'new', 'judge', Path(tmp) / 'call'))
            self.assertEqual(budget.summary()['physical_requests'], 32)
            self.assertEqual(opt.request_count(tmp), 32)

    def test_second_truncation_is_not_promoted_to_complete(self):
        api = AsyncMock(return_value=response('length'))
        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(opt.request_with_fallback(
                api, Gate(), None, {'messages': [], 'max_completion_tokens': 8192},
                object(), 'key', 'judge', Path(tmp)))
        self.assertEqual(api.await_count, 2)
        self.assertEqual(result['choices'][0]['finish_reason'], 'length')

    def test_429_during_fallback_retries_same_input_with_unique_keys(self):
        class Limited(Exception):
            status_code = 429
        calls, gate = [], Gate()
        async def api(config, wire, budget, key, stage, folder, **kw):
            calls.append((wire, key))
            if len(calls) == 1:
                return response('length')
            if len(calls) == 2:
                raise Limited('try again in 1s')
            return response()
        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(opt.request_with_fallback(
                api, gate, None, {'messages': [], 'max_completion_tokens': 8192},
                object(), 'key', 'judge', Path(tmp)))
        self.assertEqual(len(set(k for _, k in calls)), 3)
        self.assertIs(calls[1][0], calls[2][0])
        self.assertEqual(len(gate.backoffs), 1)
        self.assertEqual(result['choices'][0]['finish_reason'], 'stop')

    def test_valid_zero_partial_and_one_are_never_retried(self):
        for score in (0, 0.3, 1):
            self.assertEqual(opt.disposition({'score': score}, [{}] * 5, 32), 'valid')

    def test_retry_categories_and_attempt_limit(self):
        for error in ('query_round_budget_exhausted', 'required_evidence_incomplete:R01',
                      'ValueError: judge_output_invalid_after_repair:unknown_details_mismatch',
                      'APIConnectionError: connection lost'):
            record = {'score': None, 'errors': [error]}
            self.assertEqual(opt.disposition(record, [{}], 18), 'retry_pending')
            self.assertEqual(opt.disposition(record, [{}, {}], 19), 'held_assessment_retry_limit')
            self.assertEqual(opt.disposition(record, [{}], 32), 'held_request_budget')

    def test_nonretryable_failures_do_not_loop(self):
        record = {'score': None, 'errors': []}
        for state in ('rubric_error', 'invalid_sample'):
            self.assertEqual(opt.disposition(record, [{}], 3, {'evaluation_status': state}), 'held_' + state)
        self.assertEqual(opt.disposition({'score': None, 'errors': ['evidence_seal_changed']},
                                         [{}], 4), 'held_requires_inspection')
        self.assertEqual(opt.disposition({'score': None, 'errors': ['BudgetExceeded: global_budget_exhausted']},
                                         [{}], 4), 'held_request_budget')

    def test_attempt_history_is_preserved_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = {'score': None, 'errors': ['query_round_budget_exhausted']}
            new = {'score': 0, 'errors': []}
            opt.archive_record(tmp, old)
            opt.archive_record(tmp, old)
            opt.archive_record(tmp, new)
            values = [json.loads(p.read_text()) for p in (Path(tmp) / 'recovery_history').glob('*.json')]
            self.assertEqual(len(values), 2)
            self.assertIn(old, values)
            self.assertIn(new, values)

    def test_progress_counts_invalid_as_remaining(self):
        records = {'a': {'score': 0}, 'b': {'score': None}, 'c': {'score': 0.3}}
        status = opt.progress(records, [{'reason': 'held'}], {'total': 10, 'original_scored': 4})
        self.assertEqual(status['completed'], 2)
        self.assertEqual(status['attempts_completed'], 3)
        self.assertEqual(status['remaining'], 4)
        self.assertEqual(status['total_validation_scored'], 6)

    def test_inventory_requeues_invalid_and_preserves_valid_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repair, run = root / 'repair', root / 'run'
            opt.r.write_json(repair / 'validation/candidates.json', {'candidate': {}})
            items = []
            for i in range(3):
                source = root / ('source' + str(i))
                result = {'execution_integrity': 'valid', 'terminal_state_status': 'trusted'}
                opt.r.write_json(source / 'episode_result.json', result)
                item = {'task_instance_id': str(i), 'path': str(source)}
                items.append(item)
                key = opt.r.digest(['candidate', str(i), str(source)])[:24]
                job = run / 'judge_jobs' / key
                if i < 2:
                    target = job / 'episode'
                    score = 0 if i == 0 else None
                    opt.r.write_json(target / 'episode_result.json', {**result, 'test_score': score})
                    opt.r.write_json(job / 'recovery.json', {'episode': str(target), 'score': score,
                                                          'errors': ['query_round_budget_exhausted'] if i else []})
                    opt.r.write_json(job / 'attempts.json', [{}])
            opt.r.write_json(repair / 'validation/candidate/manifest.json', {'candidates': items})
            with patch.object(opt, 'RUN', run), patch.object(opt.r, 'REPAIRS', repair), \
                    patch.object(opt.r, 'evaluation_score', side_effect=lambda result: result.get('test_score')):
                pending, records, held, counts = opt.inventory()
            self.assertEqual({row[1]['task_instance_id'] for row in pending}, {'1', '2'})
            self.assertEqual(len(records), 2)
            self.assertFalse(held)
            self.assertEqual(opt.progress(records, held, counts)['remaining'], 2)


if __name__ == '__main__':
    unittest.main()
