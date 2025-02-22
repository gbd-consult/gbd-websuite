import json

import gws


class Error(gws.Error):
    pass


def from_path(path: str):
    """Converts a json file to a python dictionary.

    Args:
        path: Path to json file.

    Returns:
        A Python object.

    Raises:
        ``Exception``: If the given json is incorrect.
    """

    try:
        with open(path, 'rb') as fp:
            s = fp.read()
            return json.loads(s.decode('utf8'))
    except Exception as exc:
        raise Error() from exc


def from_string(s: str):
    """Converts a json string to a python dictionary.

    Args:
        s: Json string.

    Returns:
        A Python object.

    Raises:
        ``Error``: If the given json is incorrect.
    """

    if not s.strip():
        return {}
    try:
        return json.loads(s)
    except Exception as exc:
        raise Error() from exc


def to_path(path: str, x, pretty: bool = False, ensure_ascii: bool = True, default=None):
    """Converts a dictionary to a json file.

    Args:
        path: Destination of the json file.
        x: The dict to convert.
        pretty: If true then the json key-value pairs get ordered and correct indentation is used.
        ensure_ascii: If true non ASCII characters will be escaped. Else those characters will not be escaped.
        default: A function that should return a serializable version of obj or raise TypeError.
                The default simply raises TypeError.
    """

    s = to_string(x, pretty=pretty, ensure_ascii=ensure_ascii, default=default)
    try:
        gws.u.write_file_b(path, s.encode('utf8'))
    except Exception as exc:
        raise Error() from exc


def to_string(x, pretty: bool = False, ensure_ascii: bool = True, default=None) -> str:
    """Converts a dictionary to a json string.

    Args:
        x: The dict to convert.
        pretty: If true then the json key-value pairs get ordered and correct indentation is used.
        ensure_ascii: If true non ASCII characters will be escaped. Else those characters will not be escaped.
        default: A function that should return a serializable version of obj or raise TypeError.
                The default simply raises TypeError.
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
    """Converts a dictionary to a pretty json string.

        Args:
            x: The dict to convert.
            ensure_ascii: If true non ASCII characters will be escaped. Else those characters will not be escaped.
            default: A function that should return a serializable version of obj or raise TypeError.
                    The default simply raises TypeError.
        """

    return to_string(x, pretty=True, ensure_ascii=ensure_ascii, default=default)


def _json_default(x):
    try:
        return vars(x)
    except TypeError:
        return str(x)
