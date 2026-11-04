"""Quietly publish the five-batch bank after both independent runners release locks.

This watcher never dispatches a model request or resumes a stopped runner.
"""
import json
import time
import traceback
from pathlib import Path

import rubric_luna_batch04 as b


def complete(directory):
    path = directory/'completion.json'
    return path.exists() and b.read(path).get('all_frozen_tasks_terminal') is True


def lock_free(directory):
    try:
        with b.legacy.locked(directory/'.runner.lock',blocking=False):
            return True
    except BlockingIOError:
        return False


def save(status, **details):
    b.snapshot(b.RUN/'bank_supervisor_status.json',dict(at=b.now(),status=status,**details))


def main():
    b.configure()
    save('waiting_for_both_batches')
    while True:
        if not complete(b.RUN):
            if lock_free(b.RUN) and (b.RUN/'production_summary.json').exists():
                summary = b.read(b.RUN/'production_summary.json')
                save('batch04_stopped_partial',summary=summary)
                return 2
        elif complete(b.BATCH03) and lock_free(b.BATCH03) and lock_free(b.RUN):
            try:
                from rubric_luna_bank_v2 import publish
                manifest = publish()
            except BlockingIOError:
                pass
            except RuntimeError as exc:
                if str(exc).startswith(('batch_not_complete:','batch_request_still_held:')):
                    pass
                else:
                    save('publication_failed',error=repr(exc),traceback=traceback.format_exc())
                    raise
            except Exception as exc:
                save('publication_failed',error=repr(exc),traceback=traceback.format_exc())
                raise
            else:
                save('published',total_questions=manifest['total_questions'],passed=manifest['passed'],
                     bank_manifest_ref=str(b.REGISTRY/'bank_manifest.json'))
                print(json.dumps(dict(status='published',total_questions=manifest['total_questions'],
                                      passed=manifest['passed']),ensure_ascii=False),flush=True)
                return 0
        time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main())
