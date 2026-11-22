import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_awm.contracts import RunConfig, TaskSpec
from harness_awm.tracking import SwanLabTracker, check_connection, numeric_metrics, run_config


@pytest.fixture(autouse=True)
def isolated_env():
    # Never use developer credentials or contact SwanLab from tests.
    with patch.dict(os.environ, {}, clear=True), patch("harness_awm.tracking.load_dotenv"):
        yield


def test_unconfigured_and_disabled_do_not_import_sdk(tmp_path):
    with patch.dict(sys.modules, {"swanlab": None}):
        tracker = SwanLabTracker(tmp_path)
        tracker.log({"loss": 1}, step=0)
        tracker.finish()
        assert tracker.mode == "disabled"
        os.environ["SWANLAB_API_KEY"] = "fixture-key"
        assert SwanLabTracker(tmp_path, mode="disabled").run is None


def test_online_requires_key_without_interactive_prompt(tmp_path):
    with pytest.raises(ValueError, match="SWANLAB_API_KEY"):
        SwanLabTracker(tmp_path, mode="online")


def test_scalar_metrics_preserve_unknown_and_reject_nonfinite():
    assert numeric_metrics({"loss": 1.2, "reward": None, "unknown": float("nan"),
                            "infinite": float("inf"), "secret": "private",
                            "episodes": [{"answer": "private"}], "valid": {"count": 2, "ok": True}}) == {
        "loss": 1.2, "valid/count": 2, "valid/ok": 1}
    config = run_config(RunConfig(), "evaluate")
    assert "actor_api" not in config and "runtime_python" not in config
    assert config["stage"] == "evaluate"


def test_online_project_metrics_and_finalization(tmp_path):
    sdk = MagicMock()
    sdk.init.return_value.url = "https://swanlab.cn/example/run"
    with patch.dict(sys.modules, {"swanlab": sdk}), patch.dict(os.environ, {
        "SWANLAB_API_KEY": "fixture-key", "SWANLAB_PROJECT": "test-project",
        "SWANLAB_WORKSPACE": "test-workspace",
    }):
        result = check_connection(tmp_path)
    assert result["url"] == "https://swanlab.cn/example/run"
    assert sdk.init.call_args.kwargs["project"] == "test-project"
    assert sdk.init.call_args.kwargs["workspace"] == "test-workspace"
    assert sdk.init.call_args.kwargs["public"] is False
    assert "fixture-key" not in str(sdk.init.call_args)
    sdk.init.return_value.log.assert_called_once_with({"connection/check": 1}, step=0)
    sdk.finish.assert_called_once_with(state="success", error=None)


def test_offline_does_not_access_cloud_url(tmp_path):
    sdk = MagicMock()
    sdk.init.return_value = SimpleNamespace(log=MagicMock())
    with patch.dict(sys.modules, {"swanlab": sdk}):
        assert check_connection(tmp_path, "offline")["url"] is None
    assert sdk.init.call_args.kwargs["mode"] == "offline"


def test_log_failure_does_not_interrupt_work_and_marks_run_failed(tmp_path):
    sdk = MagicMock()
    sdk.init.return_value.log.side_effect = RuntimeError("private-provider-error")
    with patch.dict(sys.modules, {"swanlab": sdk}):
        tracker = SwanLabTracker(tmp_path, mode="offline")
        with pytest.warns(RuntimeWarning, match="local run evidence") as captured:
            tracker.log({"loss": 1}, step=0)
        assert "private-provider-error" not in str(captured[0].message)
        tracker.log({"loss": 2}, step=1)
        tracker.finish()
    assert sdk.init.return_value.log.call_count == 1
    assert sdk.finish.call_args.kwargs["state"] == "crashed"


@pytest.mark.parametrize("crash", [False, True])
def test_collect_stage_reports_progress_and_closes_tracker(tmp_path, crash):
    from harness_awm.stages import run_stage
    task = TaskSpec.model_construct(task_instance_id="task", scenario="fixture", native_task_id=0)
    scripts = tmp_path / "scripts.json"
    scripts.write_text('{"*": []}')
    runner = AsyncMock(side_effect=RuntimeError("fixture failure") if crash else None)
    with patch("harness_awm.stages.load_registry", return_value=([task], {})), \
         patch("harness_awm.stages.official_hashes", return_value={}), \
         patch("harness_awm.stages.EpisodeRunner.run", runner), \
         patch("harness_awm.stages.metrics", return_value={"valid_successes": 0}), \
         patch("harness_awm.stages.SwanLabTracker") as tracker:
        report = asyncio.run(run_stage("collect", tmp_path, tmp_path / "run", RunConfig(), scripts))
    assert report == {"valid_successes": 0}
    progress = tracker.return_value.log.call_args_list[0]
    assert progress.args[0]["progress/completed_candidates"] == 1
    assert progress.args[0]["progress/failed_candidates"] == int(crash)
    assert progress.kwargs["step"] == 1
    tracker.return_value.finish.assert_called_once_with(failed=crash)
