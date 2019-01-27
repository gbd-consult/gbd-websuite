import random
import threading
import os
import time
import pickle
import sys

from . import const, log


def exit(code=255):
    sys.exit(code)


def get(src, key, default=None):
    if not src:
        return default

    if isinstance(key, str):
        if '[' in key:
            key = key.replace('[', '.')
            key = key.replace(']', '.')
        key = key.split('.')

    try:
        return _get(src, key)
    except (KeyError, IndexError, AttributeError):
        return default


def extend(*args, **kwargs):
    """Merge dicts together, overwriting existing keys"""

    d = {}
    for a in args:
        d.update(as_dict(a))
    d.update(kwargs)
    return d


def assign(obj, *args, **kwargs):
    """Asssign object props from dicts"""

    d = extend(*args, **kwargs)
    for k, v in d.items():
        setattr(obj, k, v)
    return obj


def defaults(*args, **kwargs):
    """Merge dicts together, DO NOT overwrite existing keys"""

    return extend(*reversed(args + (kwargs,)))


def pick(src, keys):
    src = as_dict(src)
    return {k: src[k] for k in as_list(keys) if k in src}


def compact(src):
    if isinstance(src, dict):
        return {k: v for k, v in src.items() if v is not None}
    return [v for v in src if v is not None]


def strip(src):
    """Remove empty stuff from a nested dict/list structure"""

    def _s(x):
        if isinstance(x, (str, bytes, bytearray)):
            x = x.strip()
        return None if is_empty(x) else x

    if isinstance(src, dict):
        src = {k: _s(v) for k, v in src.items()}
    else:
        src = [_s(v) for v in src]
    return compact(src)


def is_empty(x):
    if x is None:
        return True
    try:
        return len(x) == 0
    except TypeError:
        pass
    try:
        return not vars(x)
    except TypeError:
        pass
    return False


def as_int(x):
    try:
        return int(x)
    except:
        return 0


def as_str(x, encodings=None):
    if isinstance(x, str):
        return x
    if not _is_bytes(x):
        return str(x)
    if encodings:
        for enc in encodings:
            try:
                return x.decode(encoding=enc, errors='strict')
            except UnicodeDecodeError:
                pass
    return x.decode(encoding='utf-8', errors='ignore')


def as_bytes(x):
    if _is_bytes(x):
        return x
    if not isinstance(x, str):
        x = str(x)
    return x.encode('utf8')


def as_list(x, sep=','):
    if not x:
        return []
    if _is_list(x):
        return x
    if _is_bytes(x):
        x = as_str(x)
    if isinstance(x, str):
        return strip(x.split(sep))
    if _is_atom(x):
        return [x]
    return []


def as_dict(x):
    if isinstance(x, dict):
        return x
    if hasattr(x, 'as_dict'):
        return x.as_dict()
    return {}


def as_uid(x):
    x = as_str(x).lower().strip().translate(_uid_de_trans)
    return ''.join(c if c in _uid_safe else '_' for c in x)


def as_query_string(x):
    """
        Encode a dict or a seq of pairs as a query string, 
        convert everything to utf8, join lists with comma.
        Returns a percent-encoded str.
    """

    p = []

    if isinstance(x, dict):
        x = x.items()
    elif not _is_list(x):
        x = vars(x).items()

    for k, v in sorted(x):
        k = _qs_quote(k)
        v = _qs_quote(v)
        p.append(k + b'=' + v)
    return (b'&'.join(p)).decode('ascii')


def lines(txt, comment=None):
    for s in txt.splitlines():
        if comment:
            s = s.split(comment)[0]
        s = s.strip()
        if s:
            yield s


def random_string(length):
    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    r = random.SystemRandom()
    return ''.join(r.choice(a) for _ in range(length))


class cached_property:
    def __init__(self, fn):
        self._fn = fn

    def __get__(self, obj, objtype=None):
        value = self._fn(obj)
        setattr(obj, self._fn.__name__, value)
        return value


def get_global(key, init):
    global _global_vars, global_lock

    if key in _global_vars:
        return _global_vars[key]
    with global_lock:
        if key in _global_vars:
            return _global_vars[key]
        _global_vars[key] = init()
        return _global_vars[key]


def set_global(key, value):
    global _global_vars, global_lock

    with global_lock:
        _global_vars[key] = value
        return _global_vars[key]


def get_cached_object(key, init, max_age):
    global global_lock

    with global_lock:
        return _get_cached_object(key, init, max_age)


####################################################################################################


def _is_atom(x):
    return isinstance(x, (int, float, bool, str, bytes, bytearray))


def _is_list(x):
    return isinstance(x, (tuple, list))


def _is_bytes(x):
    return isinstance(x, (bytes, bytearray))
    # @TODO how to handle bytes-alikes?
    # return hasattr(x, 'decode')


def _is_dict(x):
    return isinstance(x, dict)


def _get(src, keys):
    for k in keys:
        if isinstance(src, dict):
            src = src[k]
        elif _is_list(src):
            src = src[int(k)]
        else:
            src = getattr(src, k)
    return src


def _map_rec(src, fn):
    if _is_atom(src):
        return src

    if _is_list(src):
        d = []
        for v in src:
            if fn:
                v = fn(v)
            if v is not None:
                d.append(_map_rec(v, fn))
        return d

    if isinstance(src, dict):
        d = {}
        for k, v in src.items():
            if fn:
                v = fn(v)
            if v is not None:
                d[k] = _map_rec(v, fn)
        return d

    try:
        s2 = list(src)
    except TypeError:
        return src

    return _map_rec(s2, fn)


_uid_de_trans = {
    ord('ä'): 'ae',
    ord('ö'): 'oe',
    ord('ü'): 'ue',
    ord('ß'): 'ss',
}

_uid_safe = 'abcdefghijklmnopqrstuvwxyz0123456789_'

_qs_safe = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-'
_qs_map = {
    c: bytes([c]) if c in _qs_safe else b'%%%02X' % c
    for c in range(256)
}


def _qs_quote(x):
    return b''.join(_qs_map[c] for c in _qs_bytes(x))


def _qs_bytes(x):
    if _is_bytes(x):
        return x
    if isinstance(x, str):
        return x.encode('utf8')
    if x is True:
        return b'true'
    if x is False:
        return b'false'
    try:
        return b','.join(_qs_bytes(y) for y in x)
    except TypeError:
        return str(x).encode('utf8')


global_lock = threading.RLock()
_global_vars = {}


def _get_cached_object(key, init, max_age):
    path = const.OBJECT_CACHE_DIR + '/' + key
    obj = None

    try:
        age = time.time() - os.stat(path).st_mtime
    except:
        age = -1

    if 0 <= age < max_age:
        try:
            with open(path, 'rb') as fp:
                obj = pickle.load(fp)
        except:
            log.exception()
            pass

        if obj:
            log.debug(f'FOUND CACHED OBJECT {key!r}')
            return obj

        # something went wrong, invalidate the cache
        try:
            os.unlink(path)
        except:
            pass

    obj = init()

    if obj:
        with open(path, 'wb') as fp:
            pickle.dump(obj, fp)

    return obj
