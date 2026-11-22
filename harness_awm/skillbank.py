"""Frozen, evidence-backed procedural memory for inference (no weight updates)."""
import math
import re
from collections import Counter
from pathlib import Path

from .io import canonical, digest, file_hash, read_json, write_json

VERSION = "awm-skillbank-v1"
GUIDANCE = (
    "These are optional procedures abstracted from other training episodes, not current facts. "
    "Use only steps needed by the current user request. Read current tool schemas before calling. "
    "Obtain all identifiers and values from the current request or observations. "
    "A past successful workflow does not prove completeness or success here; check every requested outcome."
)
STOPWORDS = set("a an the to of in on for and or with from by is are be as this that it please using use".split())


def terms(text):
    return Counter(w for w in re.findall(r"[a-z]+|[\u4e00-\u9fff]", text.lower())
                   if w not in STOPWORDS and (len(w) > 1 or ord(w[0]) > 127))


class SkillBank:
    def __init__(self, config):
        self.config = config
        # Read bytes once: hash and parsed data must identify the same snapshot.
        import hashlib
        import json
        raw = Path(config.bank_path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != config.bank_sha256:
            raise ValueError("skill_bank_hash_mismatch")
        self.data = json.loads(raw)
        if self.data.get("schema_version") != VERSION or self.data.get("source_split") != "grpo_train":
            raise ValueError("unsupported_or_nontraining_skill_bank")
        self.skills = self.data["skills"]
        ids = [s["skill_id"] for s in self.skills]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate_skill_ids")
        for s in self.skills:
            if s["kind"] not in {"workflow", "failure_lesson"} or not s["evidence"]:
                raise ValueError("invalid_skill_evidence")
            if any(e.get("split") != "train" for e in s["evidence"]):
                raise ValueError("nontraining_skill_evidence")
        self.documents = {s["skill_id"]: terms(s["search_text"]) for s in self.skills}
        self.df = Counter(w for d in self.documents.values() for w in d)

    def retrieve(self, task, tools):
        """Deterministic lexical retrieval, gated by actual environment and tools."""
        query = terms(task.instruction)
        names = {t["name"] for t in tools}
        ranked = []
        for skill in self.skills:
            if skill["scenario"] is not None and skill["scenario"] != task.scenario:
                continue
            if skill["tool_schema_hash"] is not None and skill["tool_schema_hash"] != task.tool_schema_hash:
                continue
            if not set(skill["tools"]) <= names:
                continue
            # Never retrieve an episode's own distilled memory.
            if any(e["task_instance_id"] == task.task_instance_id for e in skill["evidence"]):
                continue
            doc = self.documents[skill["skill_id"]]
            overlap = query.keys() & doc.keys()
            score = sum(math.log(1 + len(self.skills) / self.df[w]) for w in overlap)
            score /= math.sqrt(max(1, len(doc)))
            if score <= 0:
                continue
            ranked.append((score, skill))
        ranked.sort(key=lambda x: (-x[0], x[1]["skill_id"]))
        selected = []
        # Prefer relevant workflows; at most one general failure reminder.
        for score, skill in ranked:
            if skill["kind"] == "failure_lesson" and any(s["kind"] == "failure_lesson" for s in selected):
                continue
            visible = {k: skill[k] for k in ("skill_id", "kind", "when_to_apply", "procedure")}
            candidate = [*selected, {**visible, "score": round(score, 6)}]
            if len(canonical(self.payload(candidate)).encode()) > self.config.max_context_bytes:
                continue
            selected = candidate
            if len(selected) == self.config.top_k:
                break
        return selected

    @staticmethod
    def payload(selected):
        return {"source": "harness", "skill_guidance": GUIDANCE,
                "skills": [{k: v for k, v in s.items() if k != "score"} for s in selected]}

    def inject(self, task, tools, folder, recorder):
        selected = self.retrieve(task, tools)
        message = {"role": "user", "content": canonical(self.payload(selected))} if selected else None
        write_json(Path(folder) / "skill_selection.json", {
            "schema_version": "awm-skill-selection-v1", "bank_sha256": self.sha256,
            "retrieval": "lexical-idf-v1", "task_instance_id": task.task_instance_id,
            "query_hash": digest(task.instruction), "selected": selected,
            "injected_bytes": len(message["content"].encode()) if message else 0,
            "message_hash": digest(message) if message else None})
        if message:
            recorder.emit("skills_injected", "model_visible", 0, bank_sha256=self.sha256,
                          skill_ids=[s["skill_id"] for s in selected],
                          input_ref=recorder.blob(message, "model_visible"))
        return message
