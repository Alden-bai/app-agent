"""Run deterministic recipes against real MCP tools, retaining raw evidence."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from .assets import scenario_asset
from .common import AWM, ROOT, backup, digest, now, save_json
from .oracle import resolve


def initialize(run_dir, scenario, target):
    schema = scenario_asset(run_dir, "gen_db.jsonl", scenario)["db_schema"]
    sample = scenario_asset(run_dir, "gen_sample.jsonl", scenario)["sample_data"]
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("Refusing to overwrite an initial DB")
    errors = []
    with sqlite3.connect(target) as db:
        for table in schema["tables"]:
            for sql in [table["ddl"], *table.get("indexes", []), *table.get("examples", [])]:
                try:
                    db.execute(sql)
                except sqlite3.Error as e:
                    errors.append({"phase": "schema", "table": table["name"], "sql": sql, "error": str(e)})
        db.execute("PRAGMA foreign_keys=ON")
        for table in sample["tables"]:
            for sql in table["insert_statements"]:
                try:
                    db.execute(sql)
                except sqlite3.Error as e:
                    errors.append({"phase": "sample", "table": table["table_name"], "sql": sql, "error": str(e)})
        fk = db.execute("PRAGMA foreign_key_check").fetchall()
        integrity = db.execute("PRAGMA integrity_check").fetchall()
    report = {"created_at": now(), "errors": errors, "foreign_key_check": fk, "integrity_check": integrity,
              "database_sha256": digest(target)}
    save_json(target.with_suffix(".initialization.json"), report)
    if errors or fk or integrity != [("ok",)]:
        raise RuntimeError("Official initialization failed; inspect initial.initialization.json")
    return report


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_script(run_dir, scenario, work_db, folder, port):
    # Use the official code adapter, but avoid its same-path final snapshot copy.
    sys.path.insert(0, str(AWM)) if str(AWM) not in sys.path else None
    from awm.core.server import Config, _generate_server_code
    env = scenario_asset(run_dir, "gen_envs.jsonl", scenario)
    import ast
    aliases = {}
    for node in ast.parse(env["full_code"]).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    for keyword in decorator.keywords:
                        if keyword.arg == "operation_id" and isinstance(keyword.value, ast.Constant):
                            aliases[node.name] = keyword.value.value
    save_json(folder / "tool_aliases.json", aliases)
    envfile = folder / "environment.jsonl"
    envfile.write_text(json.dumps(env) + "\n")
    config = Config(scenario=scenario, envs_load_path=str(envfile), db_path=str(work_db),
                    port=port, temp_server_path=str(folder / "server.py"))
    return _generate_server_code(config, str(work_db))


@asynccontextmanager
async def session(url):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    async with streamablehttp_client(url, timeout=timedelta(seconds=30),
                                    sse_read_timeout=timedelta(seconds=60)) as (read, write, _):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=60)) as client:
            await client.initialize()
            yield client


def unpack_result(result):
    if result.structuredContent is not None:
        return result.structuredContent
    texts = [item.text for item in result.content if item.type == "text"]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except ValueError:
            return texts[0]
    return texts


def select_tool(tools, name):
    exact = [t.name for t in tools if t.name == name]
    candidates = exact or [t.name for t in tools if t.name.startswith(name + "_")]
    if len(candidates) != 1:
        raise ValueError(f"Tool {name!r}: expected one match, got {candidates}")
    return candidates[0]


async def execute_case(run_dir, task, case, initial, folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    backup(initial, folder / "initial.db")
    work = folder / "working.db"
    backup(initial, work)
    port = free_port()
    server_log = (folder / "server.log").open("w")
    process = None
    trace, results, failure, answer = [], [], None, None
    try:
        script = server_script(run_dir, task["scenario"], work, folder, port)
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "TZ")}
        env.update({"HOST": "127.0.0.1", "PORT": str(port), "DATABASE_PATH": f"sqlite:///{work}"})
        process = subprocess.Popen([sys.executable, script], cwd=folder, env=env,
                                   stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        import httpx
        async with httpx.AsyncClient(timeout=1, trust_env=False) as http:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"MCP server exited with {process.returncode}; inspect server.log")
                try:
                    if (await http.get(f"http://127.0.0.1:{port}/openapi.json")).status_code == 200:
                        break
                except (httpx.HTTPError, OSError):
                    pass
                await asyncio.sleep(0.25)
            else:
                raise TimeoutError("MCP server startup exceeded 30 seconds")
        async with session(f"http://127.0.0.1:{port}/mcp") as client:
            tools = (await client.list_tools()).tools
            save_json(folder / "tools.json", [t.model_dump(mode="json") for t in tools])
            trace.append({"iteration": 1, "role": "assistant", "content": "Discover available tools.",
                          "tool_calls": [{"id": "discover", "name": "list_tools", "arguments": {}}],
                          "tool_response": {"tool_call_id": "discover", "content": json.dumps([t.name for t in tools])}})
            for index, step in enumerate(case["calls"]):
                from .common import read_json
                aliases = read_json(folder / "tool_aliases.json")
                name = select_tool(tools, aliases.get(step["tool"], step["tool"]))
                arguments = resolve(step["arguments"], work, results)
                # fastapi-mcp flattens a single HTTP body into MCP arguments.
                # Adapt only when discovery confirms that wrapper is absent.
                properties = next(t for t in tools if t.name == name).inputSchema.get("properties", {})
                for wrapper in ("body", "payload"):
                    if wrapper in arguments and wrapper not in properties and isinstance(arguments[wrapper], dict):
                        body = arguments.pop(wrapper)
                        if set(body) & set(arguments):
                            raise ValueError("Body/query argument name collision")
                        arguments.update(body)
                required = next(t for t in tools if t.name == name).inputSchema.get("required", [])
                arguments = {k: v for k, v in arguments.items() if v is not None or k in required}
                started = now()
                result = await client.call_tool(name, arguments)
                plain = unpack_result(result)
                results.append(plain)
                tc = {"id": f"call-{index}", "name": "call_tool", "arguments": {"tool_name": name, "arguments": json.dumps(arguments)}}
                trace.append({"iteration": index + 2, "role": "assistant", "content": "Execute reference recipe step.",
                              "tool_calls": [tc], "tool_response": {"tool_call_id": tc["id"], "content": json.dumps(plain, ensure_ascii=False)},
                              "started_at": started, "finished_at": now(), "is_error": result.isError})
                save_json(folder / "trajectory.partial.json", trace)
                save_json(folder / f"response_{index}.json", result.model_dump(mode="json"))
                if result.isError and any(signal in str(plain) for signal in ("Internal Server Error", "500 Server Error", "503 Service")):
                    raise RuntimeError(f"Environment server error at {name}; inspect response_{index}.json")
                if result.isError and case["expected"]:
                    raise RuntimeError(f"Reference tool {name} returned isError; inspect response_{index}.json")
            answer = resolve(case["answer"], work, results)
            if not isinstance(answer, str):
                answer = json.dumps(answer, ensure_ascii=False)
            trace.append({"iteration": len(trace) + 1, "role": "assistant", "content": answer,
                          "tool_calls": [], "is_final": True})
    except Exception as e:
        import traceback
        failure = {"type": type(e).__name__, "error": str(e),
                   "traceback": "".join(traceback.format_exception(e))}
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        server_log.close()
        if work.exists():
            backup(work, folder / "final.db")
        payload = {**task, "origin": "deterministic official MCP reference recipe", "case": case["name"],
                   "trajectory": trace, "messages": [{"role": "assistant", "content": answer or ""}],
                   "final_answer": answer, "failure": failure, "finished_at": now()}
        save_json(folder / "trajectory.json", payload)
    return payload
