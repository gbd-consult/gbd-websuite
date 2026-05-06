"""Coverage tool for server-type docstrings and ux-strings.

Run from the repository root:

    python -m gws.spec.generator.coverage [--lang de] [--threshold 80]

The tool runs the spec generator end-to-end and prints a per-module table
with three coverage metrics:

    classDoc   - share of CLASS types that have a non-empty `doc`
    propDoc    - share of PROPERTY types that have a non-empty `doc`
    uxStrings  - share of CLASSes that have at least a `purpose` entry in
                 ``uxStrings[<lang>]``

Exits with status 1 when any metric falls below ``--threshold`` (in
percent). Useful as a CI guardrail.
"""

import sys
from collections import defaultdict
from typing import Optional

from . import main, base
from ..core import c

USAGE = """
GWS spec coverage
~~~~~~~~~~~~~~~~~

    python -m gws.spec.generator.coverage [options]

Options:

    --lang <code>        - Locale to evaluate uxStrings against (default: de)
    --threshold <pct>    - Exit with code 1 if any metric in any module
                           falls below this percentage (default: 0, no gate)
    --manifest <path>    - Manifest used by the generator
    -h, --help           - Print this help and exit
"""


def _parse_args(argv: list[str]) -> dict:
    args: dict = {'lang': 'de', 'threshold': 0.0, 'manifest': ''}
    it = iter(argv)
    for a in it:
        if a in ('-h', '--help'):
            args['help'] = True
        elif a == '--lang':
            args['lang'] = next(it)
        elif a == '--threshold':
            args['threshold'] = float(next(it))
        elif a == '--manifest':
            args['manifest'] = next(it)
        else:
            raise SystemExit(f'unknown argument {a!r}')
    return args


def _module_of(typ) -> str:
    """Map a Type to a coarse module key for grouping."""
    if typ.modName:
        return typ.modName
    name = typ.name or typ.uid or '<unknown>'
    parts = name.split('.')
    return '.'.join(parts[: min(3, len(parts) - 1)] or parts)


def _is_property(typ) -> bool:
    return typ.c == c.PROPERTY


def _is_class(typ) -> bool:
    return typ.c == c.CLASS


def collect_metrics(spec_data, lang: str) -> dict[str, dict[str, tuple[int, int]]]:
    """Return ``{module: {metric: (covered, total)}}``."""
    ux = (spec_data.uxStrings or {}).get(lang) or {}
    en_ux = (spec_data.uxStrings or {}).get('en') or {}

    counters: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {'classDoc': [0, 0], 'propDoc': [0, 0], 'uxStrings': [0, 0]}
    )

    for typ in spec_data.serverTypes:
        mod = _module_of(typ)
        bucket = counters[mod]

        if _is_class(typ):
            bucket['classDoc'][1] += 1
            if (typ.doc or '').strip():
                bucket['classDoc'][0] += 1

            bucket['uxStrings'][1] += 1
            entry = ux.get(typ.name) or en_ux.get(typ.name) or {}
            if entry.get('purpose') or entry.get('label'):
                bucket['uxStrings'][0] += 1

        elif _is_property(typ):
            bucket['propDoc'][1] += 1
            if (typ.doc or '').strip():
                bucket['propDoc'][0] += 1

    return {mod: {k: tuple(v) for k, v in metrics.items()} for mod, metrics in counters.items()}


def _pct(covered: int, total: int) -> float:
    return 100.0 * covered / total if total else 100.0


def _fmt_cell(covered: int, total: int) -> str:
    if total == 0:
        return '   -   '
    return f'{_pct(covered, total):5.1f}% ({covered}/{total})'


def render_table(metrics: dict, lang: str) -> str:
    headers = ['module', 'classDoc', 'propDoc', f'uxStrings[{lang}]']
    rows = [headers]
    for mod in sorted(metrics):
        m = metrics[mod]
        rows.append(
            [
                mod,
                _fmt_cell(*m['classDoc']),
                _fmt_cell(*m['propDoc']),
                _fmt_cell(*m['uxStrings']),
            ]
        )

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    sep = '  '
    lines = []
    for r in rows:
        lines.append(sep.join(cell.ljust(widths[i]) for i, cell in enumerate(r)))
    lines.insert(1, sep.join('-' * w for w in widths))
    return '\n'.join(lines)


def below_threshold(metrics: dict, threshold: float) -> list[str]:
    """Return a list of '<module>.<metric>' offenders below the threshold."""
    offenders = []
    for mod, m in metrics.items():
        for name, (cov, total) in m.items():
            if total and _pct(cov, total) < threshold:
                offenders.append(f'{mod}.{name}: {_pct(cov, total):.1f}%')
    return offenders


def run(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(list(argv if argv is not None else sys.argv[1:]))
    if args.get('help'):
        sys.stdout.write(USAGE)
        return 0

    base.log.set_level('WARNING')
    spec_data = main.generate(manifest_path=args['manifest'] or '')

    metrics = collect_metrics(spec_data, args['lang'])
    sys.stdout.write(render_table(metrics, args['lang']) + '\n')

    threshold = float(args['threshold'])
    if threshold > 0:
        offenders = below_threshold(metrics, threshold)
        if offenders:
            sys.stdout.write(f'\n{len(offenders)} metric(s) below {threshold:.1f}%:\n')
            for o in offenders:
                sys.stdout.write(f'  - {o}\n')
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(run())
