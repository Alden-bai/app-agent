"""Run native veRL updates at lr=0 without writing large probe checkpoints."""
import argparse
import json
from pathlib import Path

from harness_awm.training.verl.sft import compose_sft_config, prepare, load_plan
from harness_awm.training.verl.sft_compat import install_sdpa_padding, verify_sft_runtime

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--tokens', type=int, required=True)
p.add_argument('--steps', type=int, default=3)
a = p.parse_args()
args = argparse.Namespace(model=Path('models/Qwen3.5-2B'), data=Path('data/training_qwen35_2b_v1'),
    output=a.output, gpus=1, global_batch_size=64, epochs=2, max_steps=a.steps, max_length=32768,
    learning_rate=1e-5, warmup_ratio=.03, seed=20261106, max_tokens_per_gpu=a.tokens)
prepare(args)
verify_sft_runtime()
install_sdpa_padding()
from harness_awm.training.verl.sft_runtime import AWMSFTTrainer
from verl.utils.distributed import initialize_global_process_group, destroy_global_process_group
config = compose_sft_config(load_plan(a.output))
config.optim.lr = 0.0
config.trainer.logger = ['console']
initialize_global_process_group()
try:
    trainer = AWMSFTTrainer(config)
    trainer.ckpt_handler.save_checkpoint = lambda **kwargs: None
    trainer.fit()
    (a.output / 'probe_complete.json').write_text(json.dumps({'steps':a.steps, 'lr':0, 'checkpoint_saved':False}))
finally:
    destroy_global_process_group()
