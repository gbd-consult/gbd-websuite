"""Translate OGC API query parameters to CQL2-Text filters."""

import datetime

from . import parser
from .parser import ParseError

_OPEN_BOUNDS = {'', '..'}


def from_datetime_param(value: str, name: str) -> str:
    """Translate the ``datetime`` parameter of OGC API Features to a CQL2-Text filter.

    The value is an instant or an interval, see OGC 17-069r4, 7.15.4. As an extension,
    dates and timestamps without a time zone are accepted, a date stands for the whole day::

        2021-06-15T10:00:00Z        name = TIMESTAMP('2021-06-15T10:00:00Z')
        2021-06-15                  name = DATE('2021-06-15')
        2021-06-01/2021-06-30       name >= DATE('2021-06-01') AND name <= DATE('2021-06-30')
        2021-06-01/..               name >= DATE('2021-06-01')

    Args:
        value: The parameter value.
        name: The temporal property to filter on.

    Returns:
        A CQL2-Text expression, empty if the value does not restrict anything.

    Raises:
        ParseError: If the value is invalid.
    """

    prop = '"' + name.replace('"', '""') + '"'
    parts = value.strip().split('/')

    conds = []
    if len(parts) == 1:
        if parts[0]:
            conds.append(f'{prop} = {_instant(parts[0], value)}')
    elif len(parts) == 2:
        lo, hi = parts[0].strip(), parts[1].strip()
        if lo not in _OPEN_BOUNDS:
            conds.append(f'{prop} >= {_instant(lo, value)}')
        if hi not in _OPEN_BOUNDS:
            conds.append(f'{prop} <= {_instant(hi, value)}')
    else:
        raise ParseError(f'invalid datetime {value!r}')

    return ' AND '.join(conds)


def _instant(s: str, value: str) -> str:
    s = s.strip()

    try:
        v = parser.parse_datetime(s)
    except ParseError:
        raise ParseError(f'invalid datetime {value!r}')

    if isinstance(v, datetime.datetime):
        return f"TIMESTAMP('{s}')"
    return f"DATE('{s}')"
