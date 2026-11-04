"""Execute unchanged released verifier through official runtime in a subprocess."""
import json
import sys
from pathlib import Path

from .common import AWM, read_json, save_json


def main():
    sys.path.insert(0, str(AWM))
    from awm.core.verify import run_verifier
    request = read_json(sys.argv[1])
    verdict, details = run_verifier(request["entry"], request["mode"], request["initial"],
                                   request["final"], request.get("answer"))
    save_json(sys.argv[2], {"verdict": verdict, "details": details})


if __name__ == "__main__":
    main()
