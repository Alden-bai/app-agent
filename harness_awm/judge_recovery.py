"""Bounded, accounted retries of a Judge request, never of an actor/tool action."""
import asyncio
from email.utils import parsedate_to_datetime
import math
from pathlib import Path
import random
import time

from .io import write_json

JUDGE_STAGES = {"judge", "judge_v2", "judge_v2_review", "trajectory_quality", "trajectory_quality_review"}


def retryable(error):
    from openai import APIConnectionError
    status = getattr(error, "status_code", None)
    if status == 429:
        return getattr(error, "code", None) in {None, "rate_limit_exceeded", "slow_down"}
    return (isinstance(error, (APIConnectionError, TimeoutError, ConnectionError))
            or status in {408, 409, 500, 502, 503, 504})


def retry_delay(error, attempt):
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    seconds = 0
    if header:
        try:
            seconds = float(header)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(header).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                pass
    return max(5 * 2 ** attempt, seconds if math.isfinite(seconds) else 0) + random.uniform(0, 1)


async def request_with_retries(send, config, kwargs, budget, key, stage, folder,
                               *, consume_hold=None, limit=None, tokens=None):
    from .backend import BudgetExceeded
    folder = Path(folder)
    attempts = getattr(config, "judge_retry_attempts", 3) if stage in JUDGE_STAGES else 1
    for index in range(attempts):
        reservation = await limit.acquire(tokens) if limit is not None else None
        suffix = f"rate_retry_{index:02d}" if limit is not None else f"retry-{index:02d}"
        call_key, call_folder = (f"{key}/{suffix}", folder / suffix) if index else (key, folder)
        try:
            value = await send(config, kwargs, budget, call_key, stage, call_folder,
                               consume_hold=consume_hold if index == 0 else None)
        except BudgetExceeded:
            if limit is not None:
                limit.refund(reservation)
            raise
        except Exception as error:
            if index + 1 >= attempts or not retryable(error):
                raise
            delay = retry_delay(error, index)
            # A long provider cooldown remains pending, rather than retrying early.
            if delay > getattr(config, "judge_retry_max_delay", 60):
                if limit is not None:
                    limit.defer(delay)
                raise
            write_json(folder / f"retry-{index:02d}.json", {
                "error_type": type(error).__name__, "status_code": getattr(error, "status_code", None),
                "next_attempt": index + 1, "delay_seconds": delay,
                "business_execution_repeated": False})
            if limit is not None:
                limit.defer(delay)
            else:
                await asyncio.sleep(delay)
        else:
            if index:
                write_json(folder / "response.json", value)
            return value
