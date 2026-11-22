from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("transformers")
from harness_awm.training.tracking import SwanLabCallback


@pytest.mark.parametrize("primary", [False, True])
def test_only_primary_process_creates_and_logs_run(primary):
    args = SimpleNamespace(output_dir="unused", learning_rate=1e-5, num_train_epochs=1,
                           max_steps=2, per_device_train_batch_size=1,
                           gradient_accumulation_steps=8, seed=42, bf16=False)
    state = SimpleNamespace(is_world_process_zero=primary, global_step=2)
    with patch("harness_awm.training.tracking.SwanLabTracker") as tracker, patch("torch.cuda.is_available", return_value=False):
        callback = SwanLabCallback()
        callback.on_train_begin(args, state, None)
        callback.on_log(args, state, None, logs={"loss": 0.5, "learning_rate": 1e-5})
        callback.finish(failed=True)
        callback.finish()
    if primary:
        tracker.assert_called_once()
        tracker.return_value.log.assert_called_once_with(
            {"train": {"loss": 0.5, "learning_rate": 1e-5}}, step=2)
        tracker.return_value.finish.assert_called_once_with(failed=True)
    else:
        tracker.assert_not_called()
