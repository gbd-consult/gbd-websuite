"""Assertions."""

import difflib
import math
import re


def xml(actual, expected):
    """Compare two XML strings, ignoring unsignificant whitespace."""

    a = norm_xml(actual)
    b = norm_xml(expected)
    assert a == b, '\n' + _diff(a, b)


def close(a, b, rel_tol=1e-9, abs_tol=0.0):
    """Compare numbers or sequences of numbers with a tolerance."""

    if isinstance(a, (int, float)):
        assert math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol), f'{a!r} != {b!r} ({rel_tol=} {abs_tol=})'
        return

    assert len(a) == len(b), f'{a!r} != {b!r} (length)'
    for n, (x, y) in enumerate(zip(a, b)):
        assert math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol), f'{a!r} != {b!r} (item {n}: {x!r} != {y!r})'


def norm_xml(s, nl=True):
    """Remove unsignificant whitespace from XML for easier comparison in tests.
    If nl is True, also put each tag on a new line for better readability.
    """

    s = re.sub(r'\s+', ' ', s.strip())
    s = re.sub(r'\s*(<|>)\s*', r'\1', s)
    if nl:
        s = s.replace('<', '\n<').replace('>', '>\n').replace('\n\n', '\n')
    return s.strip()


def _diff(actual, expected):
    return '\n'.join(
        difflib.unified_diff(
            expected.split('\n'),
            actual.split('\n'),
            'expected',
            'actual',
            lineterm='',
        )
    )
