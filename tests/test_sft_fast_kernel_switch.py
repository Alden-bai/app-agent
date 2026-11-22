import sqlite3
import pytest
from harness_awm.io import write_json
from harness_awm.training.selection import init_queue,claim,finish
from scripts.sft_fast_eval_switch import retirement_trigger,verify_benchmarks


def test_rolling_replacement_never_loses_claim_or_interrupts_owned_task(tmp_path):
    q=tmp_path/'queue.sqlite';init_queue(q,[('m','one','o1'),('m','two','o2'),('m','three','o3')])
    first=claim(q,0);retirement_trigger(q,[0,1]);finish(q,first['id'],0,'done')
    with pytest.raises(sqlite3.IntegrityError,match='planned_fast_kernel'):
        claim(q,0)
    retirement_trigger(q,[1])
    second=claim(q,0);assert second['task']=='two';finish(q,second['id'],0,'done')
    with pytest.raises(sqlite3.IntegrityError):claim(q,1)
    retirement_trigger(q,[])
    third=claim(q,1);assert third['task']=='three';finish(q,third['id'],1,'done')
    with sqlite3.connect(q) as db:
        assert db.execute("select count(*) from jobs where state='done'").fetchone()[0]==3


def test_output_difference_blocks_switch(tmp_path):
    paths=[]
    for i in range(5):
        p=tmp_path/f'{i}.json';paths.append(p)
        write_json(p,{'checkpoint':{'sha256':str(i)},'all_outputs_identical':i!=4,
            'bindings':{'f':{'fast':True}},'median_paired_wall_speedup':1.2,'pairs':[{}]})
    with pytest.raises(ValueError,match='compatibility'):
        verify_benchmarks(paths)
