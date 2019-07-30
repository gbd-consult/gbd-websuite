import json
import hashlib
import gws
import gws.types as t


class Error(gws.Error):
    pass


def from_path(path):
    try:
        with open(path, 'rb') as fp:
            s = fp.read()
            return json.loads(s.decode('utf8'))
    except Exception:
        raise Error()


def from_string(s):
    if not s.strip():
        return {}
    try:
        return json.loads(s)
    except Exception:
        raise Error()


def to_path(path, x, pretty=False):
    s = to_string(x, pretty)
    try:
        with open(path, 'wb') as fp:
            fp.write(s.encode('utf8'))
    except Exception:
        raise Error()


def to_string(x, pretty=False):
    try:
        if pretty:
            return json.dumps(x, default=gws.as_dict, indent=4, sort_keys=True)
        return json.dumps(x, default=gws.as_dict)
    except Exception as e:
        raise Error()


def to_hash(x):
    s = json.dumps(x, default=gws.as_dict, sort_keys=True)
    return hashlib.sha256(s.encode('utf8')).hexdigest()


def to_tagged_dict(x):
    keys = {}
    objects = []

    def _dict(x):
        try:
            return dict(vars(x))
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
            x.__class__.__name__,
            len(objects))

        keys[id(x)] = tag

        w = {}
        objects.append(w)
        w.update(_walk(_dict(x)))
        w['$'] = tag

        return tag

    k = _walk(x)
    return objects or [k]


def to_tagged_string(x, pretty=False):
    return to_string(to_tagged_dict(x), pretty)
