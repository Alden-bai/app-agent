"""The frozen 2,000-task GRPO experiment on the NIOFlow training pool."""
import random
from pathlib import Path

from ...io import digest, read_json

POOL_SIZE = 2746
TASK_COUNT = 2000
CANDIDATES = 4
SAVE_EVERY = 500
SELECTION = Path(__file__).resolve().parents[3] / "delivery/nioflow/grpo-2000.json"


def selected_task_ids(pool_ids, seed):
    ids = list(pool_ids)
    frozen = read_json(SELECTION)
    if (len(ids) != POOL_SIZE or len(set(ids)) != POOL_SIZE or seed != frozen["seed"]
            or digest(ids) != frozen["ordered_pool_sha256"]):
        raise ValueError("training pool or seed differs from the frozen GRPO selection")
    random.Random(seed).shuffle(ids)
    selected = ids[:TASK_COUNT]
    if selected != frozen["task_ids"]:
        raise ValueError("frozen GRPO selection does not match seeded sampling without replacement")
    return selected
