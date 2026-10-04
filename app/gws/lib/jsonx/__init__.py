"""JSON utilities.

Thin wrappers around the standard ``json`` module that read and write JSON strings and
files. Objects that are not JSON serializable are converted to their attribute dicts
(``vars``) or to strings, so that ``gws.Data`` objects can be serialized directly.
All errors are raised as ``Error``.

Example::

    s = gws.lib.jsonx.to_pretty_string({'a': 1, 'b': [1, 2]})
    d = gws.lib.jsonx.from_string(s)
    gws.lib.jsonx.to_path('/tmp/data.json', d)
"""

import json

import gws


class Error(gws.Error):
    """JSON error."""

    pass


def from_path(path: str):
    """Read a JSON file.

    Args:
        path: Path to a UTF-8 encoded JSON file.

    Returns:
        The decoded object.

    Raises:
        ``Error``: If the file cannot be read or is not valid JSON.
    """

    try:
        with open(path, 'rb') as fp:
            s = fp.read()
            return json.loads(s.decode('utf8'))
    except Exception as exc:
        raise Error() from exc


def from_string(s: str):
    """Decode a JSON string.

    Args:
        s: JSON string.

    Returns:
        The decoded object, or an empty dict if the string is empty or blank.

    Raises:
        ``Error``: If the string is not valid JSON.
    """

    if not s.strip():
        return {}
    try:
        return json.loads(s)
    except Exception as exc:
        raise Error() from exc


def to_path(path: str, x, pretty: bool = False, ensure_ascii: bool = True, default=None):
    """Write an object to a JSON file, UTF-8 encoded.

    Args:
        path: File path.
        x: Object to write.
        pretty: If ``True``, sort the keys and indent the output.
        ensure_ascii: If ``True``, escape non-ASCII characters.
        default: Function that returns a serializable version of an object that is not serializable otherwise.
            By default, objects are converted to their ``vars``, or to strings if they have none.

    Raises:
        ``Error``: If the object cannot be encoded or the file cannot be written.
    """

    s = to_string(x, pretty=pretty, ensure_ascii=ensure_ascii, default=default)
    try:
        gws.u.write_file_b(path, s.encode('utf8'))
    except Exception as exc:
        raise Error() from exc


def to_string(x, pretty: bool = False, ensure_ascii: bool = True, default=None) -> str:
    """Encode an object as a JSON string.

    Args:
        x: Object to encode.
        pretty: If ``True``, sort the keys and indent the output.
        ensure_ascii: If ``True``, escape non-ASCII characters.
        default: Function that returns a serializable version of an object that is not serializable otherwise.
            By default, objects are converted to their ``vars``, or to strings if they have none.

    Returns:
        The JSON string.

    Raises:
        ``Error``: If the object cannot be encoded.
    """

    try:
        if pretty:
            return json.dumps(
                x,
                check_circular=False,
                default=default or _json_default,
                ensure_ascii=ensure_ascii,
                indent=4,
                sort_keys=True,
            )
        return json.dumps(
            x,
            check_circular=False,
            default=default or _json_default,
            ensure_ascii=ensure_ascii,
        )
    except Exception as exc:
        raise Error() from exc


def to_pretty_string(x, ensure_ascii: bool = True, default=None) -> str:
    """Encode an object as a JSON string with sorted keys and indentation.

    Args:
        x: Object to encode.
        ensure_ascii: If ``True``, escape non-ASCII characters.
        default: Function that returns a serializable version of an object that is not serializable otherwise.
            By default, objects are converted to their ``vars``, or to strings if they have none.

    Returns:
        The JSON string.

    Raises:
        ``Error``: If the object cannot be encoded.
    """

    return to_string(x, pretty=True, ensure_ascii=ensure_ascii, default=default)


def _json_default(x):
    """Convert an object to its attribute dict, or to a string if it has none."""
    try:
        return vars(x)
    except TypeError:
        return str(x)
