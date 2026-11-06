#!/usr/bin/env python3
"""One-shot finalizer: observe the owned controller and audit when it exits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'agent-world-model/.venv/bin/python'
DEFAULT = ROOT / 'runs/environment_screen_remaining5909_20261104_v1'
FINALIZER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def alive(run, launcher):
    pid = launcher['pid']
    try:
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False
    argv = raw.decode(errors='replace').split('\x00')
    return (str(ROOT/'scripts/environment_screen_remaining.py') in argv
            and str(run) in argv and 'full' in argv)


def finalize(run):
    with (run/'finalization.log').open('ab') as log:
        result = subprocess.run([str(PYTHON), str(ROOT/'scripts/environment_screen_acceptance.py'),
                                 '--run', str(run)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    audit = json.loads((run/'acceptance.json').read_text()) if result.returncode == 0 and (run/'acceptance.json').exists() else {}
    summary = json.loads((run/'summary.json').read_text())
    complete = result.returncode == 0 and audit.get('all_required_work_complete') is True
    awaiting_master = bool(audit.get('all_scheduled_case_checks_complete') and not audit.get('master_acceptance_approved'))
    status = dict(at=datetime.now(timezone.utc).isoformat(), complete=complete,
        completed=audit.get('completed', summary['completed']), unfinished=audit.get('unfinished', summary['unfinished']),
        state='complete' if complete else 'awaiting_master_acceptance' if awaiting_master else 'stopped_with_unfinished_work',
        master_acceptance_pending=awaiting_master, audit_exit_code=result.returncode,
        audit_result_fresh=result.returncode == 0, audit_log_ref=str(run/'finalization.log'),
        reasons=summary.get('reason_counts'), evidence_errors=audit.get('structural_errors'),
        controller_status_ref=str(run/'runtime_status.json'), queue_ref=str(run/'queue.sqlite'),
        acceptance_ref=str(run/'acceptance.json'), usage_ref=str(run/'usage_detailed.json'),
        full_csv_ref=str(run/'results.full.csv'), finalizer_sha256=FINALIZER_SHA256)
    write(run/'FINAL_STATUS.json', status)
    summary['all_scheduled_case_checks_complete'] = audit.get('all_scheduled_case_checks_complete', False)
    summary['master_acceptance_approved'] = audit.get('master_acceptance_approved', False)
    summary['complete'] = complete
    write(run/'summary.json', summary)
    report_path = run/'REPORT.md'
    if report_path.exists():
        text = report_path.read_text().replace('全量完成：True', '全量完成：' + str(complete))
        if awaiting_master:
            text += '\n总控尚需验收抽检、分歧及纠正影响范围；程序计数不代表最终验收通过。\n'
        report_path.write_text(text)
    (run/'AUDIT_REPORT.md').write_text(f'''# 环境审查阶段验收\n\n全量完成：{complete}。已完成规定逐题检查 {status['completed']}，未完成 {status['unfinished']}。总控最终验收待处理：{awaiting_master}。\n\n运行状态、未完成原因见 FINAL_STATUS.json、unfinished.jsonl 和 queue.sqlite。未完成题不计入 pass/exclude/review。程序结构检查不代替总控对分歧、抽检与纠正影响范围的最终验收。\n\n完整逐题字段：results.full.csv；逐请求账本：request_usage.jsonl；验证会话用量：usage_detailed.json；主 agent 用量：master_usage.json（按明确的总控回合归属，单列，不重复合计）。\n\n并发原始核算：concurrency_audit.json；输入/引用验收：acceptance.json。\n\n该报告不将已知问题回归当作独立准确率证明，不宣称有限探测证明环境完全无缺陷，不报告无可比基线的节省率。\n''')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=DEFAULT)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    run = args.run.resolve()
    while True:
        launcher = json.loads((run/'full_launcher.json').read_text())
        running = alive(run, launcher)
        write(run/'finalizer_status.json', dict(at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
            controller_pid=launcher['pid'], controller_running=running, mode='one-shot audit on controller exit'))
        if not running:
            finalize(run)
            return
        if args.once:
            print(json.dumps(dict(controller_running=True, controller_pid=launcher['pid'])))
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
