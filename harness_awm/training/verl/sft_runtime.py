"""Use verl's native SFT trainer/engine with AWM's frozen response-only data."""
from verl.trainer.sft_trainer import SFTTrainer

from .sft_data import EncodedSFTDataset, LengthGroupedSampler, verified_rows


class AWMSFTTrainer(SFTTrainer):
    def _init_engine(self):
        super()._init_engine()
        import json
        import time
        from pathlib import Path
        import torch
        from verl.utils import tensordict_utils as tu
        from ...io import write_json

        original = self.training_client.train_batch
        folder = Path(self.config.trainer.default_local_dir).parent
        self.awm_step = 0

        def tracked_train_batch(data):
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            output = original(data=data)
            torch.cuda.synchronize()
            self.awm_step += 1
            elapsed = time.monotonic() - started
            tokens = int(data["input_ids"].offsets().diff().sum())
            metrics = tu.get(output, "metrics")
            metrics.update({"perf/step_seconds": elapsed, "perf/input_tokens_per_second": tokens / elapsed,
                "memory/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "memory/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
            if self.rank == 0:
                report = {"step": self.awm_step, "planned_steps": self.total_training_steps,
                          "input_tokens": tokens, "metrics": metrics}
                # The native worker returns scalar metrics after DP reduction.
                report = json.loads(json.dumps(report, default=lambda x: x.item()))
                write_json(folder / "progress.json", report)
                with (folder / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(report) + "\n")
            return output

        self.training_client.train_batch = tracked_train_batch

    def _build_dataset(self):
        samples, _ = verified_rows(self.config.awm.data, self.config.model.path, self.config.data.max_length)
        self.train_dataset = EncodedSFTDataset(samples, self.model_config.tokenizer.pad_token_id)
        self.val_dataset = None

    def _build_dataloader(self):
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.utils.dataset.dataset_utils import SFTTensorCollator
        world_size = self.engine.get_data_parallel_size()
        self.global_batch_size = self.config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // world_size
        self.train_sampler = LengthGroupedSampler(
            [len(s["input_ids"]) for s in self.train_dataset.samples], self.global_batch_size,
            rank=self.engine.get_data_parallel_rank(), world_size=world_size, seed=self.config.trainer.seed)
        self.collate_fn = SFTTensorCollator("no_padding")
        self.train_dataloader = StatefulDataLoader(self.train_dataset,
            batch_size=self.train_batch_size_per_dp, sampler=self.train_sampler,
            collate_fn=self.collate_fn, num_workers=0, drop_last=False)
        self.val_dataloader = None


def run_worker(config):
    import os
    import torch
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    limit = config.awm.get("gpu_memory_limit_gib", 0)
    if limit:
        total = torch.cuda.get_device_properties(local_rank).total_memory
        fraction = limit * 2**30 / total
        if not 0 < fraction <= 1:
            raise ValueError("GPU memory limit exceeds device capacity")
        torch.cuda.set_per_process_memory_fraction(fraction, local_rank)
    from .sft_compat import install_sdpa_padding
    install_sdpa_padding()
    from verl.utils.distributed import initialize_global_process_group, destroy_global_process_group
    initialize_global_process_group()
    try:
        trainer = AWMSFTTrainer(config)
        trainer.fit()
    finally:
        destroy_global_process_group()
