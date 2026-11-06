"""Selection must not hide technical faults or turn uncertainty into model failures."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('bank3000',Path(__file__).resolve().parents[1]/'scripts/execute_bank3000.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.r = {'execution':{'terminal_state_status':'trusted','execution_integrity':'valid','termination_phase':'model_request'},
            'signals':{'environment_ready':True,'tool_errors':[]},
            'evaluation':{'evaluation_valid':True,'task_status':'failed'},
            'attribution':{'failure_source':'model_failure','affects_outcome':True,'reason':'wrong object despite available correct tool result','evidence_ids':['E1']}}

    def test_valid_model_failure_is_kept(self):
        self.assertEqual(module.classify(self.r)['label'],'valid_model_failure')

    def test_missing_attribution_does_not_imply_model_failure(self):
        self.r.pop('attribution')
        self.assertEqual(module.classify(self.r)['disposition'],'review')

    def test_normal_business_error_can_still_be_model_failure(self):
        self.r['signals']['tool_errors']=[{'turn':2}]
        self.assertEqual(module.classify(self.r)['disposition'],'keep')

    def test_recovered_tool_error_does_not_discard_success(self):
        self.r['signals']['tool_errors']=[{'turn':2}]
        self.r['evaluation']['task_status']='confirmed_pass'
        self.assertEqual(module.classify(self.r)['label'],'valid_success')

    def test_untrusted_state_never_kept_even_if_judge_passes(self):
        self.r['execution']['execution_integrity']='outcome_unknown'
        self.r['evaluation']['task_status']='confirmed_pass'
        self.assertEqual(module.classify(self.r)['disposition'],'review')

    def test_fault_requires_causal_impact(self):
        self.r['attribution']['failure_source']='tool_failure'
        self.r['attribution']['affects_outcome']=None
        self.assertEqual(module.classify(self.r)['disposition'],'review')
        self.r['attribution']['affects_outcome']=True
        self.assertEqual(module.classify(self.r)['disposition'],'quarantine')
        self.assertEqual(module.classify(self.r)['confidence'],'model_assisted')

    def test_judge_error_is_not_model_or_environment_failure(self):
        self.r['evaluation']={'evaluation_valid':False,'evaluation_status':'evaluation_error'}
        self.assertEqual(module.classify(self.r)['label'],'evaluation_failure')

    def test_api_failure_is_separate(self):
        self.r['actor_api_errors']=['episode/model_turns/0001/error.json']
        self.assertEqual(module.classify(self.r)['label'],'api_failure')

    def test_initialization_failure_is_mechanical(self):
        self.r['execution'].update(termination_phase='preparing',terminal_state_status='unavailable',error='environment_exited_during_startup')
        self.r['signals']['environment_ready']=False
        self.assertEqual(module.classify(self.r)['label'],'environment_failure')

    def test_preflight_failure_preserves_denominator(self):
        self.assertEqual(module.classify({'preflight_errors':['missing snapshot']})['label'],'input_failure')

    def test_unknown_outcome_cannot_be_kept_as_model_failure(self):
        self.r['evaluation']['task_status']='provisional_pass'
        self.assertEqual(module.classify(self.r)['disposition'],'review')

    def test_failure_without_trajectory_is_not_a_valid_failure(self):
        self.assertEqual(module.classify({})['disposition'],'review')


if __name__=='__main__': unittest.main()
