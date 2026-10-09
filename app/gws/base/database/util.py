"""Database utilities."""

from typing import Optional

import gws
import gws.lib.sa as sa


def text_search_clause(column, value: Optional[str], tso: Optional[gws.TextSearchOptions]) -> Optional[sa.ColumnElement]:
    """Create a where clause that matches a column against a search string.

    The value is stripped. For the ``any``, ``begin`` and ``end`` search types,
    ``LIKE`` wildcards in the value match literally. For the ``like`` type, the value
    is used as a ``LIKE`` pattern, with ``\\`` as the escape character.

    Args:
        column: Column expression to match.
        value: Search string.
        tso: Text search options. Without options, the value is matched exactly.

    Returns:
        A clause, or ``None`` if the value is empty or shorter than the minimum length.
    """

    if value is None:
        return

    value = str(value).strip()
    if not value:
        return

    if not tso:
        return column == value

    if tso.minLength and len(value) < tso.minLength:
        return

    if tso.type == gws.TextSearchType.exact:
        return column == value

    cs = tso.caseSensitive

    if tso.type == gws.TextSearchType.any:
        return column.contains(value, autoescape=True) if cs else column.icontains(value, autoescape=True)
    if tso.type == gws.TextSearchType.begin:
        return column.startswith(value, autoescape=True) if cs else column.istartswith(value, autoescape=True)
    if tso.type == gws.TextSearchType.end:
        return column.endswith(value, autoescape=True) if cs else column.iendswith(value, autoescape=True)

    return column.like(value, escape='\\') if cs else column.ilike(value, escape='\\')
