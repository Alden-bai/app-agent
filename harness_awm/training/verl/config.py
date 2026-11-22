"""Compose the pinned upstream config without importing its GPU workers."""
from pathlib import Path


def validate_turnwise_config(config):
    """Validate upstream contracts with a DP-aligned *turn* batch.

    AWM samples one task with K candidates, then expands their actual turns.
    Upstream's prompt-batch divisibility check assumes one completion per row.
    Validate an independent aligned view; never change the task loader or K.
    Actual update batches are aligned by core.turn_batch and checked separately.
    """
    import copy
    from omegaconf import open_dict
    from verl.utils.config import validate_config
    aligned = copy.deepcopy(config)
    size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    with open_dict(aligned):
        aligned.data.train_batch_size = size
        aligned.actor_rollout_ref.actor.ppo_mini_batch_size = size
    validate_config(aligned, use_reference_policy=False, use_critic=False)


def compose_config(plan):
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    directory = Path(verl.__file__).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="ppo_trainer")
    updates = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "data.train_batch_size": 1,
        "data.val_batch_size": 1,
        "data.dataloader_num_workers": 0,
        "data.shuffle": False,
        "data.seed": plan["run_config"]["seed"],
        "data.max_prompt_length": plan["run_config"]["context_limit"] - plan["run_config"]["max_output_tokens"],
        "data.max_response_length": plan["run_config"]["max_output_tokens"],
        "actor_rollout_ref.model.path": plan["model"],
        "actor_rollout_ref.model.use_remove_padding": False,
        "actor_rollout_ref.model.use_fused_kernels": False,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.override_config": {"attn_implementation": "sdpa"},
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        # Adam states are only needed during updates; release them before vLLM wake-up.
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": plan.get("optimizer_offload", True),
        "actor_rollout_ref.actor.fsdp_config.param_offload": plan.get("param_offload", False),
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.clip_ratio": plan.get("clip", 0.2),
        "actor_rollout_ref.actor.clip_ratio_low": plan.get("clip", 0.2),
        "actor_rollout_ref.actor.clip_ratio_high": plan.get("clip", 0.2),
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.optim.lr": plan["learning_rate"],
        "actor_rollout_ref.actor.optim.weight_decay": 0.01,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum",
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra", "hf_model"],
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": plan.get("rollout_tensor_parallel_size", 1),
        "actor_rollout_ref.rollout.n": plan["group_size"],
        "actor_rollout_ref.rollout.max_model_len": plan["run_config"]["context_limit"],
        "actor_rollout_ref.rollout.max_num_seqs": plan["concurrency"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": plan["gpu_memory_utilization"],
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.logprobs_mode": "raw_logprobs",
        "actor_rollout_ref.rollout.agent.agent_loop_manager_class":
            "harness_awm.training.verl.runtime.HarnessRolloutManager",
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": plan["gpus"],
        "trainer.total_epochs": max(1, plan["groups"]),
        "trainer.total_training_steps": plan.get("target_updates") or plan["groups"],
        "trainer.val_before_train": False,
        "trainer.test_freq": -1,
        "trainer.resume_mode": "disable",
        "trainer.logger": ["console"],
        "trainer.project_name": "awm-verl",
        "trainer.experiment_name": Path(plan["output"]).name,
        "trainer.default_local_dir": plan.get("checkpoint_dir", str(Path(plan["output"]) / "checkpoints")),
        "trainer.default_hdfs_dir": None,
        "reward.num_workers": 1,
        "ray_kwargs.ray_init.runtime_env.env_vars.AWM_VERL_TEXT_MODEL": "1",
        "ray_kwargs.ray_init.runtime_env.env_vars.AWM_VERL_RUNTIME_PROFILE": plan.get("runtime_profile", "pinned"),
        "ray_kwargs.ray_init.num_cpus": 8,
        "ray_kwargs.ray_init.object_store_memory": 2 * 1024**3,
        "ray_kwargs.ray_init.include_dashboard": False,
    }
    if plan.get("runtime_profile") in {"local-sdpa", "local-sdpa-cu128"}:
        from .compat import cuda_home
        updates["ray_kwargs.ray_init.runtime_env.env_vars.CUDA_HOME"] = cuda_home(plan['runtime_profile'])
    if plan.get("pilot_guards"):
        updates["ray_kwargs.ray_init.runtime_env.env_vars.AWM_PILOT_ALLOCATOR_GB"] = str(plan.get("allocator_gb", 72))
        updates["actor_rollout_ref.rollout.max_num_batched_tokens"] = 4096
    with open_dict(config):
        for key, value in updates.items():
            OmegaConf.update(config, key, value, merge=False, force_add=False)
        config.awm = {"plan": str(Path(plan["output"]) / "plan.json")}
    return config

