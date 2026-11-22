import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness_awm.backend import Budget, BudgetExceeded
from harness_awm.io import digest, file_hash, read_json, write_json
from scripts.sft_rate_recheck_inputs import HOLD_SCHEMA, REQUEST_SCHEMA, load_cohort, merge_budget


def make_budget(path, rows, holds=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(REQUEST_SCHEMA)
        db.execute(HOLD_SCHEMA)
        db.executemany('INSERT INTO requests VALUES (?,?,?,?,?,?,?,?)', rows)
        db.executemany('INSERT INTO holds VALUES (?,?)', holds)
    return path


def request(key, *, identifier=1, reserved=100, status='usage_unknown'):
    return (identifier, key, 'judge_v2', reserved, status, None, 'start', 'finish')


class BudgetMergeTest(unittest.TestCase):
    def test_deduplicates_copied_baseline_and_preserves_fields_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_budget(root / 'first.sqlite', [request('same')])
            extra = (8, 'extra', 'judge_v2_review', 17, 'response', '{"total_tokens":17}', 's', 'f')
            second = make_budget(root / 'second.sqlite', [request('same', identifier=7), extra])
            original_hashes = [file_hash(first), file_hash(second)]
            target = root / 'merged.sqlite'
            result = merge_budget([first, second, first], target)
            self.assertEqual(result['baseline'], {
                'physical_requests': 2, 'tokens_accounted_or_reserved': 117,
                'unknown_usage_requests': 1, 'held_requests': 0, 'held_tokens': 0})
            self.assertEqual(result['max_cumulative_requests'], 32)
            self.assertEqual([item['sha256'] for item in result['sources']], original_hashes)
            self.assertEqual([file_hash(first), file_hash(second)], original_hashes)
            with sqlite3.connect(target) as db:
                actual = db.execute('SELECT request_key,stage,reserved,status,usage,started,finished '
                                    'FROM requests ORDER BY id').fetchall()
            self.assertEqual(actual, [request('same')[1:], extra[1:]])
            self.assertEqual(Budget(target, 32, None).summary(), result['baseline'])
            self.assertEqual(read_json(target.with_suffix('.baseline.json')), result)

    def test_conflicting_request_is_rejected_before_destination_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = make_budget(root / 'one.sqlite', [request('same')])
            two = make_budget(root / 'two.sqlite', [request('same', reserved=101)])
            with self.assertRaisesRegex(ValueError, 'budget_request_conflict'):
                merge_budget([one, two], root / 'target.sqlite')
            self.assertFalse((root / 'target.sqlite').exists())

    def test_active_holds_and_existing_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_budget(root / 'source.sqlite', [], [('pending', 500)])
            with self.assertRaisesRegex(ValueError, 'budget_has_active_holds'):
                merge_budget([source], root / 'target.sqlite')
            before = file_hash(source)
            with self.assertRaises(FileExistsError):
                merge_budget([source], source)
            self.assertEqual(file_hash(source), before)

    def test_history_at_request_limit_still_blocks_new_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_budget(root / 'source.sqlite',
                                 [request(str(i), identifier=i + 1) for i in range(32)])
            target = root / 'target.sqlite'
            self.assertEqual(merge_budget([source], target)['baseline']['physical_requests'], 32)
            with self.assertRaises(BudgetExceeded):
                Budget(target, 32, None).reserve('new', 'judge_v2', [], 1)

    def test_baseline_precedes_publication_and_interrupted_preparation_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_budget(root / 'source.sqlite', [request('one')])
            target = root / 'target.sqlite'
            sidecar = target.with_suffix('.baseline.json')

            def interrupted_publish(*args):
                self.assertTrue(sidecar.exists())
                self.assertFalse(target.exists())
                raise OSError('simulated interruption')

            with patch('scripts.sft_rate_recheck_inputs.os.link', side_effect=interrupted_publish):
                with self.assertRaisesRegex(OSError, 'simulated interruption'):
                    merge_budget([source], target)
            baseline = read_json(sidecar)
            self.assertEqual(merge_budget([source], target), baseline)


class CohortTest(unittest.TestCase):
    def fixture(self, root, candidate, task_id):
        source = root / candidate / ('job_' + task_id) / 'episodes/t00000/a0'
        full_id = 'manifest:scenario:' + task_id
        write_json(source / 'task.json', {'task_instance_id': full_id,
                                         'scenario': 'scenario', 'native_task_id': task_id})
        make_budget(source.parents[2] / 'budget.sqlite', [request('original:' + task_id)])
        key = digest([candidate, full_id, str(source)])[:24]
        return {'candidate': candidate, 'task_id': 'scenario:' + task_id,
                'latest_failure': 'RateLimitError', 'source_episode': str(source),
                'recovery_job': str(root / 'new_jobs' / key)}

    def inventory(self, root, rows):
        path = root / 'inventory.csv'
        with path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_filters_interleaves_and_includes_all_budget_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a1, a2 = self.fixture(root, 'a', '1'), self.fixture(root, 'a', '2')
            b1 = self.fixture(root, 'b', '1')
            ignored = dict(a1, latest_failure='APIConnectionError', source_episode='/absent')
            repair = root / 'repairs/judge_jobs_v1/old'
            write_json(repair / 'recovery.json', dict(candidate='a', source_episode=a1['source_episode'],
                       task_instance_id='manifest:scenario:1'))
            old_budget = make_budget(repair / 'budget.sqlite', [request('repair')])
            new_budget = make_budget(Path(a1['recovery_job']) / 'budget.sqlite', [request('repair')])
            result = load_cohort(self.inventory(root, [a1, ignored, a2, b1]),
                                 repairs_root=root / 'repairs', expected_count=3)
            self.assertEqual([(r['candidate'], r['task_id']) for r in result],
                             [('a', 'scenario:1'), ('b', 'scenario:1'), ('a', 'scenario:2')])
            self.assertEqual(result[0]['task_instance_id'], 'manifest:scenario:1')
            self.assertEqual(result[0]['key'], Path(a1['recovery_job']).name)
            self.assertEqual(result[0]['budget_sources'], [
                str(Path(a1['source_episode']).parents[2] / 'budget.sqlite'),
                str(old_budget), str(new_budget)])

    def test_wrong_count_identity_and_duplicate_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.fixture(root, 'a', '1')
            inventory = self.inventory(root, [row])
            with self.assertRaisesRegex(ValueError, 'cohort_count_mismatch'):
                load_cohort(inventory, repairs_root=root, expected_count=1061)
            for change, error in [({'task_id': 'other:1'}, 'cohort_task_id_mismatch'),
                                  ({'recovery_job': str(root / 'wrong')}, 'cohort_recovery_key_mismatch')]:
                with self.assertRaisesRegex(ValueError, error):
                    load_cohort(self.inventory(root, [dict(row, **change)]),
                                repairs_root=root, expected_count=1)
            with self.assertRaisesRegex(ValueError, 'duplicate_cohort_key'):
                load_cohort(self.inventory(root, [row, row]), repairs_root=root, expected_count=2)

    def test_source_symlink_does_not_relocate_parent_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.fixture(root, 'a', '1')
            source = Path(row['source_episode'])
            archive = root / 'archive'
            source.rename(archive)
            source.symlink_to(archive, target_is_directory=True)
            result = load_cohort(self.inventory(root, [row]), repairs_root=root, expected_count=1)
            self.assertEqual(result[0]['budget_sources'],
                             [str(source.parents[2] / 'budget.sqlite')])


if __name__ == '__main__':
    unittest.main()
