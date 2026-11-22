from types import SimpleNamespace
import pytest
from scripts.benchmark_shared_batch import batch_decode


def policy():
    torch=pytest.importorskip('torch')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            self.generation_config=SimpleNamespace(eos_token_id=2);self.calls=0
        def prepare_inputs_for_generation(self,ids,**kw):
            mask=kw['attention_mask'];positions=kw['position_ids']
            assert positions[0,-1]==mask[0].sum()-1
            return {'input_ids':ids}
        def forward(self,input_ids):
            n=len(input_ids);logits=torch.full((n,1,10),-10.)
            for i in range(n):
                # Row 0 ends immediately. Row 1 produces a token then chat EOS.
                token=2 if i==0 else (7 if self.calls==0 else 1)
                logits[i,0,token]=10.
            self.calls+=1
            return SimpleNamespace(logits=logits,past_key_values=object())
    return Model(),SimpleNamespace(eos_token_id=1,pad_token_id=0)


def test_independent_eos_and_variable_prompt_lengths():
    model,tok=policy()
    result=batch_decode(model,tok,[[3],[4,5,6]],max_tokens=5)
    assert result['completion_ids']==[[2],[7,1]]
    assert model.calls==2


def test_replay_has_fixed_work_even_if_predictions_would_end_early():
    model,tok=policy()
    result=batch_decode(model,tok,[[3],[4,5,6]],max_tokens=5,replay=[[8,8,2],[7,1]])
    assert result['completion_ids']==[[8,8,2],[7,1]]
    assert result['predicted_ids'][0]==[2,2,2]
    assert model.calls==3


def test_replay_rejects_invalid_lengths():
    model,tok=policy()
    with pytest.raises(ValueError,match='replay'):
        batch_decode(model,tok,[[3]],max_tokens=1,replay=[[4,2]])
