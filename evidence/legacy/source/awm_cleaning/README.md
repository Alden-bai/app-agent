# Official AWM cleaning workflow

This package is separate from `awm_audit`; it never edits the original audit,
released task text, environment implementation, or verifier. The only formal
configuration is the released SQL evidence extractor plus the official AWM LLM
Judge (`deepseek-flash`, temperature 1, 4096 completion tokens, 120 seconds).
`gen_verifier.pure_code.jsonl` is neither downloaded nor a prerequisite.

The frozen population is commit
`dde80a0283fe781bdc51656bce57063dc5650213`: 1,000 environments and 10,000 tasks.
Preparation reads all seven source assets, stores exact bytes and hashes,
retains every raw task row/index, checks name normalization, and records all
missing/duplicate/mismatched bindings. A duplicate is blocked, never resolved by
choosing one record. The pilot is the exact original 100 task texts and IDs from
`audit_runs/20261021_seed20261021/manifest.json`.

Run from `/home/bjx/app-agent`, using the existing isolated dependencies:

```bash
.audit-venv/bin/python -m awm_cleaning prepare
.audit-venv/bin/python -m awm_cleaning run --scope pilot100 --max-api-requests 1000
.audit-venv/bin/python -m awm_cleaning run --scope pilot100 --resume --max-api-requests 1000
.audit-venv/bin/python -m awm_cleaning report
.audit-venv/bin/python -m awm_cleaning publish
.audit-venv/bin/python -m pytest -q awm_cleaning/tests
```

Use `--run-dir PATH` **before** the subcommand for a separate release run.
`prepare --source DIRECTORY` accepts already downloaded exact JSONL files.
Hugging Face TLS compatibility, if required by the local network, uses the
existing certificate-verifying `AWM_AUDIT_TLS_FQDN=1` adapter. MCP requires local
loopback networking; a sandbox socket denial is infrastructure uncertainty,
never a task defect. Credentials are read through the existing dotenv config;
no key is written to manifests or public releases.

Full scope is deliberately two separate choices:

```bash
.audit-venv/bin/python -m awm_cleaning run --scope all --resume \
  --max-api-requests REQUEST_CAP --authorize-all-model-budget
```

This example is documentation, not authorization to execute it. The user must
explicitly approve the full-run model budget first. The cap is cumulative across
process restarts and counts every physical request, including transport retries.
Each frozen case has exactly five new semantic slots. Only connection/timeout,
429, and 5xx errors get at most two transport retries; 401/402/403 opens the
circuit. Malformed JSON, truncation, Judge errors and `server_error` remain
invalid; `agent_error` and `incomplete` are valid refusals. No semantic retry or
best-of selection is performed. A slot with no dispatched request can resume
after an explicit budget increase; an interrupted in-flight slot remains unknown.

The serial implementation deliberately avoids concurrent shared instrumentation.
Environment checks include three complete imports and business-state digests,
real MCP discovery, JSON schema validation, and actual empty-input probes for
every exposed tool. Task-specific checks separately establish legal writes,
readback, boundary behavior, three modified-state resets and two simultaneously
live isolated services. Empty-input/schema checks do not establish all business
semantics; unresolved tool dependencies block admission.

Executor arguments accept only explicit task literals, a frozen contract-value
registry, prior tool return paths, and registered reproducible computations.
Audit-side SQL is separate. `$sql`, unannotated scalars, forged contract constants,
and future return references are rejected before execution. Existing reference
recipes are inspected as candidates, not silently upgraded: many contain hidden
DB-derived IDs and incomplete independent requirement mappings.

A complete task plugin supplies cited requirements, real recipes, independent
read-only predicates, applicability/equivalence rules, effective counterexamples,
frozen challenge cases, and a complete SQL evidence reviewer. The implemented
`reference_ecommerce.py` covers the original review-submission task
`e_commerce_10:7`; the generic pipeline independently inspects all other pilot
tasks and records exactly which evidence is still missing. This is an honest
incremental implementation: inspection of 100 tasks does not mean that all 100
have passed every certification gate. There is no generic semantic oracle that
can certify arbitrary English tasks from an old successful trajectory alone.

The SQL reviewer must independently reconstruct the original extractor output,
including object, joins, filters, aggregation, nulls and omitted rows. A
`partially_checked` result cannot trigger a certifying model call. Judge input is
only the original task, real trajectory and original SQL extraction; independent
labels, rules, references and expected outcomes stay private. Extraction executes
on DB copies and detects mutation/errors before Judge dispatch.

Admission is deterministic: an evidence-reviewed reachable hard defect yields
`reject`; every gate passing yields `keep`; everything else yields `quarantine`.
A model failure, task difficulty or old single Judge result is never a reason to
reject. Rules/cases and raw task/environment/transport evidence are hashed;
completed valid slots are immutable on resume. Certificate revisions are retained.
Changing source, dependencies or Judge configuration requires a new frozen run.

Output lives in `cleaning_runs/<release_sha>/`: full ledger/manifest;
`environment_checks.jsonl`, `tool_checks.jsonl`; private task specifications,
DBs, provenance, cases, five-slot responses and versioned certificates;
`task_certificates.jsonl`, `approved_verifiers.jsonl`; disjoint
`tasks.keep/reject/quarantine.jsonl`; `report.md`, `summary.json`, and
`cost_report.json`. Reports explicitly distinguish unprocessed tasks,
checked-insufficient tasks and upstream-blocked tasks. The full denominator is
always `keep + reject + quarantine = 10000`.

Costs include physical requests, retries, reported tokens, API elapsed time and
stored bytes, with per-task records. Full-run request/token intervals are
conditional on the observed case-count and token ranges, not a confidence
interval or a currency estimate. Provider prices are not invented.

`publish` writes only keep task instances, a single fixed SQL+LLM configuration,
selected official assets, original initial DBs, actual AWM runtime source, exact
runtime dependency metadata, artifact checksums and a standalone public loader.
It does not publish private audit answers/rules/recipes. Empty publication is
explicitly marked empty. SFT collection, GRPO training and evaluation are outside
this cleaning workflow.
