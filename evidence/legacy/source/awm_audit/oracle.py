"""Independent rules: this module never imports or reads released verifiers."""
from __future__ import annotations

import copy
import importlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .common import digest, query


class NeedsReview(Exception):
    """A reference cannot be established; this is NOT proof of unsolvability."""


@dataclass
class Rule:
    name: str
    sql: str
    params: list = field(default_factory=list)


@dataclass
class TaskSpec:
    calls: list
    rules: list[Rule]
    mutations: list = field(default_factory=list)
    answer: object = "Requested actions executed; see tool results."
    answer_expected: object = None
    answer_path: list = field(default_factory=list)
    answer_columns: list = field(default_factory=list)
    answer_ordered: bool = True
    answer_subset: bool = False
    notes: str = ""
    status: str = "ready"
    evidence: object = None
    required_tools: list = field(default_factory=list)

    def frozen(self):
        return asdict(self)

    def cases(self):
        out = [{"name": "positive", "expected": True, "calls": self.calls, "answer": self.answer}]
        # Every label is subsequently checked by the oracle; an unchanged state
        # that already satisfies the task will not become a false negative.
        out.append({"name": "no_action", "expected": False, "calls": [], "answer": "No actions performed."})
        for i, mutation in enumerate(self.mutations):
            calls, answer = copy.deepcopy(self.calls), copy.deepcopy(self.answer)
            if mutation.get("kind") == "omit":
                calls = calls[:mutation["index"]]
            elif mutation.get("kind") == "answer":
                answer = mutation["value"]
            else:
                target = calls[mutation["index"]]["arguments"]
                keys = mutation["path"]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = mutation["value"]
            out.append({"name": mutation.get("name", f"wrong_{i}"), "expected": False,
                        "calls": calls, "answer": answer})
        # Do not inflate coverage by counting identical plans multiple times.
        seen, names, unique = set(), set(), []
        for c in out:
            # For state-only requirements, different closing prose does not
            # make the same tool plan a distinct negative test.
            h = digest([c["calls"], c["answer"] if self.answer_expected is not None else None])
            if h not in seen:
                seen.add(h)
                base = re.sub(r"[^A-Za-z0-9_.-]+", "_", c["name"])[:120].strip("._") or "case"
                c["name"] = base if base not in names else base + "_" + h[:10]
                names.add(c["name"])
                unique.append(c)
        return unique


def call(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


def sqlref(sql, *, params=(), many=False, column=None):
    return {"$sql": sql, "params": list(params), "many": many, "column": column}


def resultref(index=-1):
    return {"$result": index}


def mutate(index, path, value, name=None):
    return {"index": index, "path": path.split(".") if isinstance(path, str) else path,
            "value": value, "name": name or "wrong_" + str(path)}


def one(db, sql, params=()):
    data = query(db, sql, params)
    if len(data) != 1:
        raise NeedsReview(f"Expected exactly one independent target, found {len(data)}: {sql}")
    return data[0]


def scalar(db, sql, params=()):
    return next(iter(one(db, sql, params).values()))


def row_rule(table, values, *, where="1", new=False, count=1, name=None):
    """Check all supplied columns together on the SAME row, with multiplicity."""
    terms, params = [f"({where})"], []
    for key, value in values.items():
        if isinstance(value, (list, dict)):
            # Semantic JSON equivalence, not raw whitespace comparison.
            terms.append(f"json(t.\"{key}\")=json(?)")
            params.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        elif isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) and (
                key.endswith(("_date", "_deadline")) or key == "date_of_birth"):
            terms.append(f'date(t."{key}") = ?')
            params.append(value)
        else:
            terms.append(f't."{key}" IS ?')
            params.append(value)
    if new:
        terms.append(f'NOT EXISTS (SELECT 1 FROM initial."{table}" i WHERE i.id=t.id)')
    return Rule(name or table + " required fields", f'SELECT count(*)={count} FROM "{table}" t WHERE ' + " AND ".join(terms), params)


def load_spec(scenario, idx, initial):
    module = importlib.import_module("awm_audit.rules." + scenario)
    return module.build(idx, initial)


def resolve(value, work_db, results):
    if isinstance(value, dict):
        if "$compute" in value:
            from .reference_answers import compute
            return compute(value, results)
        if "$sql" in value:
            found = query(work_db, value["$sql"], resolve(value.get("params", []), work_db, results))
            if value.get("column") is not None:
                vals = [r[value["column"]] for r in found]
            else:
                vals = [next(iter(r.values())) if len(r) == 1 else r for r in found]
            if value.get("many"):
                return vals
            if len(vals) != 1:
                raise NeedsReview(f"Runtime SQL reference needs one row; got {len(vals)}")
            return vals[0]
        if "$result" in value:
            result = copy.deepcopy(results[value["$result"]])
            if "$sort" in value:
                config = value["$sort"]
                target = result
                for key in config.get("path", []):
                    target = target[key]
                target.sort(key=lambda x: x[config["by"]], reverse=config.get("reverse", False))
            return result
        return {k: resolve(v, work_db, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, work_db, results) for v in value]
    return value


def evaluate(spec, initial_db, final_db, answer, trajectory=None):
    checks = []
    try:
        with sqlite3.connect(f"file:{Path(final_db).resolve()}?mode=ro", uri=True) as db:
            db.execute("ATTACH DATABASE ? AS initial", (f"file:{Path(initial_db).resolve()}?mode=ro",))
            db.execute("PRAGMA query_only=ON")
            for rule in spec.rules:
                values = db.execute(rule.sql, rule.params).fetchall()
                # An empty query is unknown/error, never vacuous success.
                valid = len(values) == 1 and len(values[0]) == 1 and values[0][0] in (0, 1)
                checks.append({"name": rule.name, "sql": rule.sql, "params": rule.params,
                               "value": values, "passed": bool(values[0][0]) if valid else None})
        if spec.answer_expected is not None:
            value = answer
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    value = None
            for key in spec.answer_path:
                value = value.get(key) if isinstance(value, dict) else None
            if spec.answer_columns and isinstance(value, list):
                value = [{k: row.get(k) for k in spec.answer_columns} for row in value if isinstance(row, dict)]
            expected = spec.answer_expected
            if spec.answer_subset:
                value = project(value, expected)
            checks.append({"name": "requested answer", "expected": expected,
                           "actual": value, "passed": equivalent(value, expected, ordered=spec.answer_ordered)})
        if spec.required_tools:
            called = {tc.get("arguments", {}).get("tool_name") for event in (trajectory or [])
                      if not event.get("is_error") for tc in event.get("tool_calls", [])}
            for tool in spec.required_tools:
                checks.append({"name": "required API evidence: " + tool, "called": sorted(x for x in called if x),
                               "passed": tool in called or tool.removesuffix("_endpoint") in called})
        if not checks:
            return {"status": "unknown", "correct": None, "error": "No independent predicates", "checks": []}
        if any(c["passed"] is False for c in checks):
            return {"status": "valid", "correct": False, "checks": checks}
        if any(c["passed"] is None for c in checks):
            return {"status": "unknown", "correct": None, "checks": checks}
        return {"status": "valid", "correct": all(c["passed"] for c in checks), "checks": checks}
    except Exception as e:
        return {"status": "unknown", "correct": None, "error": f"{type(e).__name__}: {e}", "checks": checks}


def project(actual, expected):
    if isinstance(expected, dict) and isinstance(actual, dict):
        return {k: project(actual[k], v) for k, v in expected.items() if k in actual}
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return [project(a, e) for a, e in zip(actual, expected)]
    return actual


def equivalent(actual, expected, ordered=True):
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equivalent(actual[k], expected[k], ordered) for k in expected)
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        if not ordered:
            actual = sorted(actual, key=lambda x: json.dumps(x, sort_keys=True))
            expected = sorted(expected, key=lambda x: json.dumps(x, sort_keys=True))
        return all(equivalent(a, e, ordered) for a, e in zip(actual, expected))
    if type(actual) in (float, int) and type(expected) in (float, int):
        return abs(actual - expected) <= 1e-7 * max(1, abs(expected))
    return type(actual) is type(expected) and actual == expected
