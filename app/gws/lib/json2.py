import json

import gws


class Error(gws.Error):
    pass


def from_path(path):
    try:
        with open(path, 'rb') as fp:
            s = fp.read()
            return json.loads(s.decode('utf8'))
    except Exception as e:
        raise Error() from e


def from_string(s):
    if not s.strip():
        return {}
    try:
        return json.loads(s)
    except Exception as e:
        raise Error() from e


def to_path(path, x, pretty=False):
    s = to_string(x, pretty)
    try:
        with open(path, 'wb') as fp:
            fp.write(s.encode('utf8'))
    except Exception:
        raise Error()


def to_string(x, pretty=False, ascii=True):
    try:
        if pretty:
            return json.dumps(x, default=_json_default, indent=4, sort_keys=True, ensure_ascii=ascii)
        return json.dumps(x, default=_json_default, ensure_ascii=ascii)
    except Exception as e:
        raise Error() from e


def to_pretty_string(x, ascii=True):
    return to_string(x, pretty=True, ascii=ascii)


def to_tagged_dict(x):
    keys = {}
    objects = []

    def _dict(x):
        try:
            return {k: v for k, v in vars(x).items() if not callable(v)}
        except TypeError:
            return {}

    def _walk(x):

        if x is None or isinstance(x, (int, float, bool, str)):
            return x

        if isinstance(x, (bytes, bytearray)):
            return 'bytes(%r)' % x

        if isinstance(x, dict):
            return {str(k): _walk(v) for k, v in x.items()}

        if isinstance(x, set):
            x = list(x)

        if isinstance(x, (list, tuple)):
            return [_walk(v) for v in x]

        if id(x) in keys:
            return keys[id(x)]

        tag = '$%s.%s:%d' % (
            getattr(x, '__module__', ''),
            getattr(x, '__class__', '').__name__,
            len(objects))

        keys[id(x)] = tag

        w = {}
        objects.append(w)
        w.update(_walk(_dict(x)))
        w['$'] = tag

        return tag

    k = _walk(x)
    return objects or [k]


def to_tagged_string(x, pretty=False, ascii=True):
    return to_string(to_tagged_dict(x), pretty, ascii)


def _json_default(x):
    if gws.is_data_object(x):
        return vars(x)
    return str(x)
