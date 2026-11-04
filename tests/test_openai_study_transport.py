import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness_awm.backend import Budget, api_request
from harness_awm.contracts import APIConfig
from harness_awm.io import read_json


class OpenAITransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoning_setting_keeps_official_prompt_and_parameters(self):
        sent = []
        class Client:
            def __init__(self, **kwargs): self.chat = SimpleNamespace(completions=self)
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def create(self, **kwargs):
                sent.append(kwargs)
                return SimpleNamespace(model_dump=lambda **_: {'usage': {'total_tokens': 20}})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_OPENAI_KEY':'secret'}), patch('openai.AsyncOpenAI', Client):
            cfg = APIConfig(base_url='https://api.openai.com/v1', model='gpt-6-luna', api_key_env='TEST_OPENAI_KEY', reasoning_effort='none')
            request = {'model':cfg.model, 'messages':[{'role':'system','content':'unchanged official prompt'}], 'temperature':1.0, 'max_completion_tokens':4096}
            budget = Budget(Path(tmp)/'budget.sqlite', 1, 10000)
            await api_request(cfg, request, budget, 'judge1', 'judge', Path(tmp)/'call')
            self.assertEqual(sent[0], {**request, 'reasoning_effort':'none'})
            self.assertNotIn('reasoning_effort', request)
            self.assertNotIn('extra_body', sent[0])
            self.assertEqual(read_json(Path(tmp)/'call/request.json'), sent[0])
            self.assertEqual(budget.summary()['tokens_accounted_or_reserved'],20)
