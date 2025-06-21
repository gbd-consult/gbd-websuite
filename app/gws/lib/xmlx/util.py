import datetime


def atom_to_string(s) -> tuple[str, bool]:
    """Return a string representation of a primitive value.

    Returns:
        A tuple containing the string representation and a boolean indicating if the conversion was successful.
    """

    if s is None:
        return '', True

    if isinstance(s, str):
        return s, True

    if isinstance(s, (int, float, bool)):
        return str(s).lower(), True

    if isinstance(s, datetime.datetime):
        return s.strftime('%Y-%m-%dT%H:%M:%S'), True

    if isinstance(s, datetime.date):
        return s.strftime('%Y-%m-%d'), True

    return '', False


def escape_text(s: str) -> str:
    """Escape special characters in a string for XML."""

    s = s.replace('&', '&amp;')
    s = s.replace('>', '&gt;')
    s = s.replace('<', '&lt;')
    return s


def escape_attribute(s: str) -> str:
    """Escape special characters in a string for XML attributes."""

    s = s.replace('&', '&amp;')
    s = s.replace('"', '&quot;')
    s = s.replace('>', '&gt;')
    s = s.replace('<', '&lt;')
    return s
