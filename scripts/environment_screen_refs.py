"""Resolve case-local evidence files and verify line or JSON location annotations."""
import json
from pathlib import Path
import re


def resolve_ref(case, ref):
    if not isinstance(ref, str) or not ref:
        raise ValueError('evidence reference must be a nonempty string')
    case = Path(case).resolve()
    direct = (case / ref).resolve()
    if direct.is_relative_to(case) and direct.is_file():
        return direct
    name, marker, fragment = ref.partition('#')
    lines = None
    match = re.fullmatch(r'(.*):(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)', name)
    if match:
        name, lines = match.groups()
    path = (case / name).resolve()
    if not path.is_relative_to(case) or not path.is_file():
        raise ValueError('missing/out-of-case evidence: ' + ref)
    if marker and re.fullmatch(r'L\d+(?:-L?\d+)?(?:,L?\d+(?:-L?\d+)?)*', fragment):
        if lines:
            raise ValueError('duplicate line annotations: ' + ref)
        lines = fragment.replace('L', '')
        fragment = ''
    if lines:
        with path.open() as stream:
            total = sum(1 for _ in stream)
        for segment in lines.split(','):
            bounds = segment.strip().split('-')
            start, end = int(bounds[0]), int(bounds[-1])
            if not 1 <= start <= end <= total:
                raise ValueError('nonexistent evidence line/range: ' + ref)
    if marker and fragment:
        value = json.loads(path.read_text())
        if fragment.startswith('/'):
            tokens = [p.replace('~1', '/').replace('~0', '~') for p in fragment[1:].split('/')]
        else:
            expression = fragment.removeprefix('$.')
            if expression.startswith('$['):
                expression = expression[1:]
            tokens, pos = [], 0
            pattern = re.compile(r'(?:\.?([A-Za-z_][\w-]*))|(?:\[(\d+)\])')
            while pos < len(expression):
                match = pattern.match(expression, pos)
                if not match:
                    raise ValueError('unsupported JSON evidence location: ' + ref)
                tokens.append(match.group(1) if match.group(1) is not None else match.group(2))
                pos = match.end()
        candidates = [tokens]
        if fragment.startswith('/') and '[' in fragment:
            # Common hybrid notation #/queries[0] still has to resolve to real content.
            expression = fragment[1:].replace('/', '.')
            alternate, pos = [], 0
            pattern = re.compile(r'(?:\.?([A-Za-z_][\w-]*))|(?:\[(\d+)\])')
            while pos < len(expression):
                match = pattern.match(expression, pos)
                if not match:
                    break
                alternate.append(match.group(1) if match.group(1) is not None else match.group(2))
                pos = match.end()
            if pos == len(expression):
                candidates.append(alternate)
        for candidate in candidates:
            node = value
            try:
                for token in candidate:
                    if isinstance(node, list):
                        if not str(token).isdigit():
                            raise KeyError(token)
                        node = node[int(token)]
                    elif isinstance(node, dict):
                        node = node[token]
                    else:
                        raise KeyError(token)
                break
            except (KeyError, IndexError, TypeError):
                continue
        else:
            raise ValueError('nonexistent JSON evidence location: ' + ref)
    return path
