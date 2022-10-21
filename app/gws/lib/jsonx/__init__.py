import json

import gws


class Error(gws.Error):
    pass


def from_path(path):
    try:
        with open(path, 'rb') as fp:
            s = fp.read()
            return json.loads(s.decode('utf8'))
    except Exception as exc:
        raise Error() from exc


def from_string(s):
    if not s.strip():
        return {}
    try:
        return json.loads(s)
    except Exception as exc:
        raise Error() from exc


def to_path(path, x, pretty=False, ensure_ascii=True, default=None):
    s = to_string(x, pretty=pretty, ensure_ascii=ensure_ascii, default=default)
    try:
        with open(path, 'wb') as fp:
            fp.write(s.encode('utf8'))
    except Exception as exc:
        raise Error() from exc


def to_string(x, pretty=False, ensure_ascii=True, default=None):
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


def to_pretty_string(x, ensure_ascii=True, default=None):
    return to_string(x, pretty=True, ensure_ascii=ensure_ascii, default=default)


def _json_default(x):
    try:
        return vars(x)
    except TypeError:
        return str(x)
