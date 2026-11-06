import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sft_review', Path(__file__).resolve().parents[1]/'scripts/sft_review_pilot.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReviewPilotTests(unittest.TestCase):
    def test_fifty_reproducible_nonoverlapping_strata(self):
        pool = []
        for i in range(100):
            pool.append(dict(id=f't{i:04d}', scenario=f's{i%60}', calls=0 if i<2 else 2,
                             error_signal=2<=i<30, turn_count=25 if 30<=i<50 else 8))
        selected, gaps = m.select(pool)
        again, _ = m.select(list(reversed(pool)))
        self.assertEqual(selected, again)
        self.assertEqual(len({r['id'] for r in selected}), 50)
        self.assertEqual(gaps, [])
        self.assertEqual(dict(m.Counter(r['stratum'] for r in selected)), dict(m.QUOTAS))

    def test_exact_inputs_survive_dedup_and_context_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = Path(tmp)
            a = {'role':'user','content':'task'}
            b = {'role':'assistant','content':'bad call'}
            c = {'role':'user','content':'error'}
            inputs = [[a], [a,b,c], [a, {'role':'user','content':'compacted ledger'}]]
            for i, messages in enumerate(inputs,1):
                m.write(episode/'model_turns'/f'{i:04d}'/'input.json', {'messages':messages})
                m.write(episode/'model_turns'/f'{i:04d}'/'output.json', {'content':f'out{i}','finish_reason':'stop'})
            pack, _ = m.make_pack(episode)
            restored = [[pack['messages'][r] for r in t['input_refs']] for t in pack['turns']]
            self.assertEqual(inputs, restored)
            self.assertNotIn(b,restored[2])

    def fixture(self, root):
        folder = root/'materials/t0001'
        m.write(folder/'pack.json', {'messages':{},'turns':[{'turn_id':1,'output':'bad'}, {'turn_id':2,'output':'fixed'}]})
        m.write(folder/'checks.json', {'errors':[]})
        ref = str(folder/'pack.json')+'#/turns/0/output'
        record = {'id':'t0001','decision':'keep','reason':'recovered','coverage_turn_ids':[1,2],
                  'evidence_refs':[ref],'exclude_loss_turn_ids':[1],
                  'issues':[{'tag':'invalid_argument','turn_ids':[1],'evidence_refs':[ref],'reason':'invalid'}]}
        return folder,record

    def test_invalid_turn_and_reference_and_missing_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,r = self.fixture(Path(tmp))
            m.validate(r,'t0001',folder)
            for field, val in [('exclude_loss_turn_ids',[99]), ('coverage_turn_ids',[1]),
                               ('evidence_refs',[str(folder/'pack.json')+'#/turns/50/output']),
                               ('issues',[])]:
                bad=copy.deepcopy(r);bad[field]=val
                with self.assertRaises((ValueError,IndexError)):
                    m.validate(bad,'t0001',folder)

    def test_processing_failure_and_resume_are_not_data_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,r=self.fixture(root)
            manifest={'selected':[{'id':'t0001'}], 'second_review_ids':[],
                      'config':{'max_attempts':3},'materials':{'t0001':{'a':'hash'}}}
            with patch.object(m,'OUT',root),patch.object(m,'verified_manifest',return_value=manifest):
                self.assertEqual(m.status()['primary']['missing'],['t0001'])
                result=root/'reviews/primary/t0001.json'
                m.write(result,{'broken':True})
                self.assertIn('t0001',m.status()['primary']['invalid'])
                self.assertEqual(m.status()['primary']['done'],[])
                m.write(result,r)
                self.assertEqual(m.status()['primary']['done'],['t0001'])
                with self.assertRaisesRegex(ValueError,'already completed'):
                    m.start_job('t0001','primary','duplicate')
                changed=copy.deepcopy(r);changed['reason']='changed'
                m.write(result,changed)
                self.assertIn('stale cache',m.status()['primary']['invalid']['t0001'])

    def test_attempt_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            manifest={'selected':[{'id':'t0001'}], 'second_review_ids':[],
                      'config':{'max_attempts':3},'materials':{'t0001':{}}}
            with patch.object(m,'OUT',root),patch.object(m,'verified_manifest',return_value=manifest):
                for n in range(3):m.start_job('t0001','primary',str(n))
                with self.assertRaisesRegex(ValueError,'attempt limit'):
                    m.start_job('t0001','primary','fourth')

    def test_source_hash_verification_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original.json';source.write_text('{"v":1}')
            m.write(root/'source_hashes.json',{str(source):m.sha(source)})
            m.write(root/'candidates.json',[])
            manifest={'source_hashes_sha256':m.sha(root/'source_hashes.json'),
                      'candidates_sha256':m.sha(root/'candidates.json')}
            with patch.object(m,'OUT',root),patch.object(m,'verified_manifest',return_value=manifest):
                self.assertEqual(m.verify_sources()['changed'],[])
                source.write_text('{"v":2}')
                self.assertEqual(m.verify_sources()['changed'],[str(source)])


if __name__=='__main__':unittest.main()
