"""References must resolve to real files and actual anchored content."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from environment_screen_refs import resolve_ref


class EvidenceReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name) / 'case'
        self.case.mkdir()
        (self.case / 'code.py').write_text('one\ntwo\nthree\nfour\nfive\n')
        (self.case / 'result.json').write_text('{"before_queries":[{"rows":[[1]]}],"a/b":{"~x":2}}')

    def test_multirange_line_locations(self):
        self.assertEqual(resolve_ref(self.case, 'code.py:1,3-5'), self.case/'code.py')
        self.assertEqual(resolve_ref(self.case, 'code.py#L1-L3'), self.case/'code.py')

    def test_json_fragments_and_pointer(self):
        self.assertEqual(resolve_ref(self.case, 'result.json#before_queries[0].rows[0][0]'),self.case/'result.json')
        self.assertEqual(resolve_ref(self.case, 'result.json#/a~1b/~0x'),self.case/'result.json')
        self.assertEqual(resolve_ref(self.case, 'result.json#/before_queries[0]/rows[0]'),self.case/'result.json')

    def test_nonexistent_line_and_json_location_rejected(self):
        for ref in ['code.py:6', 'code.py:4-3', 'result.json#before_queries[1]', 'result.json#missing']:
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                resolve_ref(self.case, ref)

    def test_no_path_or_symlink_escape(self):
        (self.case.parent/'outside.json').write_text('{}')
        (self.case/'link.json').symlink_to(self.case.parent/'outside.json')
        for ref in ['../outside.json', 'link.json', 'link.json#x']:
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                resolve_ref(self.case, ref)


if __name__ == '__main__':
    unittest.main()
