"""Value conversion and escaping for XML output."""

import gws.lib.datetimex as dtx


def atom_to_string(s) -> tuple[str, bool]:
    """Convert a primitive value to a string for XML output.

    ``None`` becomes an empty string, numbers and booleans are lowercased (``true``, ``false``),
    datetimes and dates are written in ISO format.

    Args:
        s: The value.

    Returns:
        A tuple of the string and a flag that is ``False`` if the value is not a primitive value.
    """

    if s is None:
        return '', True

    if isinstance(s, str):
        return s, True

    if isinstance(s, (int, float, bool)):
        return str(s).lower(), True

    if isinstance(s, dtx.dt.datetime):
        return dtx.to_iso_string(s, with_tz=':'), True

    if isinstance(s, dtx.dt.date):
        return dtx.to_iso_date_string(s), True

    return '', False


def escape_text(s: str) -> str:
    """Escape ``&``, ``<`` and ``>`` for XML text content.

    Args:
        s: The text.

    Returns:
        The escaped text.
    """

    s = s.replace('&', '&amp;')
    s = s.replace('>', '&gt;')
    s = s.replace('<', '&lt;')
    return s


def escape_attribute(s: str) -> str:
    """Escape a string for a double-quoted XML attribute value.

    Escapes ``&``, ``"``, ``<``, ``>``, tabs and line breaks.

    Args:
        s: The value.

    Returns:
        The escaped value.
    """

    s = s.replace('&', '&amp;')
    s = s.replace('"', '&quot;')
    s = s.replace('>', '&gt;')
    s = s.replace('<', '&lt;')
    s = s.replace('\t', '&#x9;')
    s = s.replace('\r', '&#xd;')
    s = s.replace('\n', '&#xa;')
    return s
