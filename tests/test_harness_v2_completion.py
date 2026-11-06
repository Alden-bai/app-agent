import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import complete_harness_v2_experiment as completion
from harness_awm.backend import Budget
from harness_awm.io import write_json


class CompletionSelectionTests(unittest.TestCase):
    def test_only_undispatched_actor_can_start_and_valid_judgments_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'; source.mkdir()
            write_json(source/'manifest.json',{})
            budget=Budget(source/'budget.sqlite',10,10000)
            tasks=[{'id':f't{i}'} for i in range(4)]
            for i in range(4):
                e={'termination_reason':'global_budget_limit' if i==0 else 'finish','tool_calls':1 if i==1 else 0}
                ev={'evaluation_valid':i==2,'errors':[] if i==2 else ['BudgetExceeded: global_budget_exhausted']}
                write_json(source/'original_budget'/f't{i}'/'result.json',{'execution':e,'evaluation':ev})
                if i!=0:
                    rid=budget.reserve(f'original_budget/t{i}/actor/1','actor',[],1)
                    budget.finish(rid,{'total_tokens':1})
            manifest={'configs':{'original_budget':{}},'tasks':tasks}
            with patch.object(completion,'load_manifest',return_value=manifest), patch.object(completion,'verify_seal',return_value={}):
                result=completion.prepare(source,Path(tmp)/'new',5000000,1000)
            self.assertEqual([(e['id'],e['mode']) for e in result['entries']],
                             [('t0','first_actor_request'),('t1','judge_only'),('t3','judge_only')])
            self.assertEqual(budget.summary()['physical_requests'],3)
            self.assertFalse((Path(tmp)/'new/budget.sqlite').exists())

    def test_preflight_only_rows_have_serializable_termination_bucket(self):
        import harness_v2_experiment as experiment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest={'configs':{'original_budget':{}},'tasks':[{'id':'t0','ready':False,
                'task_id':'missing:0','preflight_error':'missing initial state','baseline':{'task_status':'undetermined'}}]}
            write_json(root/'original_budget/t0/result.json',{'status':'preflight_blocked','error':'missing initial state'})
            with patch.object(experiment,'load_manifest',return_value=manifest):
                result=experiment.report(root)
            self.assertEqual(result['arms']['original_budget']['terminations'],{'not_executed':1})


if __name__=='__main__': unittest.main()
