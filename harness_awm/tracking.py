"""Optional SwanLab metrics shared by collection, evaluation and future training."""
import math
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

from .contracts import ROOT


def numeric_metrics(values, prefix=""):
    """Flatten finite scalar metrics; omit text, missing values and episode lists."""
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(numeric_metrics(value, name))
        elif isinstance(value, (int, float)) and math.isfinite(value):
            result[name] = int(value) if isinstance(value, bool) else value
    return result


def run_config(config, stage):
    # Do not send API endpoints, credentials, task instructions or local paths.
    fields = ("protocol_id", "purpose", "actor", "judge", "judge_protocol", "workers",
              "model_concurrency", "judge_concurrency", "candidates_per_task", "max_turns",
              "max_tool_calls", "context_limit", "max_output_tokens", "seed")
    return {"stage": stage, **{key: getattr(config, key) for key in fields},
            "actor_model": config.actor_api.model, "judge_model": config.judge_api.model}


class SwanLabTracker:
    def __init__(self, run_dir, *, config=None, mode=None):
        load_dotenv(ROOT / ".env", override=False)
        self.sdk = self.run = None
        self.log_failed = False
        self.mode = mode or os.environ.get("SWANLAB_MODE") or (
            "online" if os.environ.get("SWANLAB_API_KEY", "").strip() else "disabled")
        if self.mode == "disabled":
            return
        if self.mode not in {"online", "offline", "local"}:
            raise ValueError("SWANLAB_MODE must be online, offline, local or disabled")
        if self.mode == "online" and not os.environ.get("SWANLAB_API_KEY", "").strip():
            raise ValueError("SwanLab online mode requires SWANLAB_API_KEY in .env or environment")
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError("Install SwanLab with: python -m pip install -e '.[tracking]'") from exc
        run_dir = Path(run_dir).resolve()
        self.sdk = swanlab
        self.run = swanlab.init(
            project=os.environ.get("SWANLAB_PROJECT") or "awm-agent",
            workspace=os.environ.get("SWANLAB_WORKSPACE") or None,
            name=os.environ.get("SWANLAB_EXPERIMENT_NAME") or run_dir.name,
            log_dir=str(run_dir / "swanlog"), mode=self.mode, public=False,
            on_online_error="raise", config=config or {},
            settings=swanlab.Settings(
                interactive=False,
                terminal=swanlab.Settings.Terminal(proxy_type="none"),
                probe=swanlab.Settings.Probe(git=False, runtime=False, requirements=False),
            ),
        )

    @property
    def url(self):
        return self.run.url if self.run is not None and self.mode == "online" else None

    def log(self, values, *, step, strict=False):
        if self.run is None or self.log_failed:
            return
        data = numeric_metrics(values)
        if not data:
            return
        try:
            self.run.log(data, step=step)
        except Exception:
            if strict:
                raise
            self.log_failed = True
            warnings.warn("SwanLab metric logging failed; local run evidence remains available.",
                          RuntimeWarning, stacklevel=2)

    def finish(self, *, failed=False, strict=False):
        if self.run is None:
            return
        try:
            crashed = failed or self.log_failed
            self.sdk.finish(state="crashed" if crashed else "success",
                            error="Harness run or metric logging failed" if crashed else None)
        except Exception:
            if strict:
                raise
            warnings.warn("SwanLab finalization failed; check local swanlog before syncing.",
                          RuntimeWarning, stacklevel=2)
        finally:
            self.run = None


def check_connection(run_dir, mode="online"):
    tracker = SwanLabTracker(run_dir, config={"stage": "connection-check"}, mode=mode)
    failed = True
    try:
        tracker.log({"connection/check": 1}, step=0, strict=True)
        result = {"mode": tracker.mode, "url": tracker.url, "run": str(Path(run_dir).resolve())}
        failed = False
        return result
    finally:
        tracker.finish(failed=failed, strict=True)
