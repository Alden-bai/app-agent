import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('correspondence',Path(__file__).resolve().parents[1]/'scripts/sft_review_correspondence.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class CorrespondenceTests(unittest.TestCase):
    def test_event_model_roundtrip_and_missing_event_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); inp=[{'role':'user','content':'task'}];output={'content':'answer','finish_reason':'stop'}
            (root/'input.json').write_text(json.dumps(inp));(root/'output.json').write_text(json.dumps(output))
            def ref(name): return {'path':name,'sha256':m.f.p.sha(root/name)}
            events=[{'seq':1,'event_type':'model_request','turn_id':1,'event_id':'a','input_ref':ref('input.json')},
                    {'seq':2,'event_type':'model_response','turn_id':1,'event_id':'b','parent_event_id':'a','raw_model_output_ref':ref('output.json')}]
            pack={'messages':{'m1':inp[0]},'turns':[{'turn_id':1,'input_refs':['m1'],'output':'answer','finish_reason':'stop','output_metadata':{}}]}
            def run():
                (root/'events.jsonl').write_text('\n'.join(map(json.dumps,events)))
                return m.check(root,pack)
            self.assertFalse(run()['errors'])
            pack['turns'][0]['output']='wrong';self.assertTrue(run()['errors'])
            pack['turns'][0]['output']='answer';events.pop();self.assertTrue(run()['errors'])


if __name__=='__main__': unittest.main()
