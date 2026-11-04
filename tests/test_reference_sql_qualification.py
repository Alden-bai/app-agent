import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from judge_accuracy100_openai_v2 import strict_sql_evidence
from judge_accuracy100_openai_v3 import query_id_allowed


class QualificationTests(unittest.TestCase):
    def test_failed_query_can_be_repaired_but_successful_evidence_is_immutable(self):
        query={'id':'q1','sql':'SELECT * FROM final.items'}
        self.assertTrue(query_id_allowed(query,{}))
        self.assertTrue(query_id_allowed(query,{'q1':{'valid':False}}))
        self.assertFalse(query_id_allowed(query,{'q1':{'valid':True}}))
        self.assertFalse(query_id_allowed({'id':'task','sql':'SELECT 1'},{'task':{'valid':True}}))

    def test_initial_and_final_cannot_be_confused(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'initial.db',Path(tmp)/'final.db'
            for p,n in [(a,1),(b,2)]:
                with sqlite3.connect(p) as db:
                    db.execute('CREATE TABLE items(id INTEGER, value TEXT)')
                    db.executemany('INSERT INTO items VALUES(?,?)',[(i,str(i)) for i in range(n)])
            def q(sql):return strict_sql_evidence(a,b,{'id':'q','sql':sql})
            self.assertEqual(q('SELECT COUNT(*) FROM initial.items')['rows'],[[1]])
            self.assertEqual(q('SELECT COUNT(*) FROM final.items')['rows'],[[2]])
            self.assertEqual(q('WITH x AS (SELECT * FROM final.items) SELECT count(*) FROM x')['rows'],[[2]])
            for sql in ['SELECT * FROM items','SELECT COUNT(*) FROM items',
                        'SELECT * FROM final.items f JOIN items i ON i.id=f.id',
                        'SELECT * FROM final.items WHERE id IN (SELECT id FROM items)']:
                result=q(sql)
                self.assertFalse(result['valid'],(sql,result))
                self.assertIn('Unqualified',result['error'])
            self.assertFalse(q('DELETE FROM final.items')['valid'])
            self.assertEqual(q('SELECT COUNT(*) FROM final.items')['rows'],[[2]])
