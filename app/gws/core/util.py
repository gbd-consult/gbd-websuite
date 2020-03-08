"""Core utilities

Most common function which are needed everywhere. These function are exported in `gws` and can be used as gws.function().
"""

import hashlib
import os
import pickle
import random
import re
import sys
import threading
import time

import gws.core.const
import gws.types as t


def exit(code: int = 255):
    """Exit the application.

    Args:
        code: Exit code.
    """

    sys.exit(code)


def get(data, key, default=None):
    """Get a nested value/attribute from data.

    Args:
        data: A dict, list or Data.
        key: A list or a dot separated string of nested keys.
        default: The default value.

    Returns:
        The value if it exists and the default otherwise.
    """

    if not data:
        return default

    if isinstance(key, str):
        key = key.split('.')

    try:
        return _get(data, key)
    except (KeyError, IndexError, AttributeError):
        return default


def merge(data, *args, **kwargs):
    """Update a dict/Data object with the values from dicts/Datas or kwargs.

    Args:
        data: A dict or a Data.
        *args: Dicts or Datas.
        **kwargs: Keyword args.

    Returns:
        A new object (dict or Data).
    """

    d = dict(as_dict(data))

    for a in args:
        d.update(as_dict(a))
    d.update(kwargs)

    if isinstance(data, dict):
        return d

    return type(data)(d)


def _is_not_empty_or_blank(x):
    if isinstance(x, (str, bytes, bytearray)):
        x = x.strip()
    return not is_empty(x)


def filter(data, fn=None):
    """Apply a filter to a collection.

    Args:
        data: A dict/Data or an iterable.
        fn: Filtering function, if omitted, blank strings and empty values are removed.

    Returns:
        A filtered object.
    """

    fn = fn or _is_not_empty_or_blank

    if isinstance(data, dict):
        return {k: v for k, v in data.items() if fn(v)}
    if hasattr(data, 'as_dict'):
        d = {k: v for k, v in data.as_dict().items() if fn(v)}
        return type(data)(d)
    return [x for x in data if fn(x)]


def _is_not_none(x):
    return x is not None


def compact(data):
    """Remove all None values from a collection."""

    return filter(data, _is_not_none)


def map(data, fn):
    """Apply a function to a collection.

    Args:
        data: A dict/Data or an iterable.
        fn: A function.

    Returns:
        A mapped object.
    """

    if isinstance(data, dict):
        return {k: fn(v) for k, v in data.items()}
    if hasattr(data, 'as_dict'):
        d = {k: fn(v) for k, v in data.as_dict().items()}
        return type(data)(d)
    return [fn(x) for x in data]


def _strip(x):
    if isinstance(x, (str, bytes, bytearray)):
        return x.strip()
    return x


def strip(data):
    """Strip all str values in a collection and remove empty values.

    Args:
        data: A dict/Data or an iterable.

    Returns:
        The stripped object.
     """

    return filter(map(data, _strip))


def is_empty(x) -> bool:
    """Check if the value is empty (None, empty list/dict/object)."""

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


def as_int(x) -> int:
    """Convert a value to an int or 0 if this fails."""

    try:
        return int(x)
    except:
        return 0


def as_float(x) -> float:
    """Convert a value to a float or 0.0 if this fails."""

    try:
        return float(x)
    except:
        return 0.0


def as_str(x, encodings: t.List[str] = None) -> str:
    """Convert a value to a string.

    Args:
        x: Value.
        encodings: A list of acceptable encodings. If the value is bytes, try each encoding,
            and return the first one which passes without errors.

    Returns:
        A string.
    """

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


def as_bytes(x) -> bytes:
    """Convert a value to bytes by converting it to string and encoding in utf8."""

    if _is_bytes(x):
        return bytes(x)
    if not isinstance(x, str):
        x = str(x)
    return x.encode('utf8')


def as_list(x, delimiter: str = ',') -> list:
    """Convert a value to a list.

    Args:
        x: A value. Is it's a string, split it by the delimiter
        delimiter:

    Returns:
        A list.
    """

    if isinstance(x, list):
        return x
    if is_empty(x):
        return []
    if _is_bytes(x):
        x = as_str(x)
    if isinstance(x, str):
        if delimiter:
            ls = [s.strip() for s in x.split(delimiter)]
            return [s for s in ls if s]
        return [x]
    if isinstance(x, (int, float, bool)):
        return [x]
    try:
        return [s for s in x]
    except TypeError:
        return []


def as_dict(x) -> dict:
    """Convert a value to a dict. If the argument provides `as_dict`, use it."""

    if isinstance(x, dict):
        return x
    if hasattr(x, 'as_dict'):
        return x.as_dict()
    return {}


_UID_DE_TRANS = {
    ord('ä'): 'ae',
    ord('ö'): 'oe',
    ord('ü'): 'ue',
    ord('ß'): 'ss',
}


def as_uid(x) -> str:
    """Convert a value to an uid (alphanumeric string)."""

    x = as_str(x).lower().strip().translate(_UID_DE_TRANS)
    x = re.sub(r'[^a-z0-9]+', '_', x)
    return x.strip('_')


def as_query_string(x) -> str:
    """Convert a dict/list to a query string.

    For each item in x, if it's a list, join it with a comma, stringify and in utf8.

    Args:
        x: Value, which can be a dict'able or a list of key,value pairs.

    Returns:
        The query string.
    """

    p = []
    items = x if _is_list(x) else as_dict(x).items()

    for k, v in sorted(items):
        k = _qs_quote(k)
        v = _qs_quote(v)
        p.append(k + b'=' + v)

    return (b'&'.join(p)).decode('ascii')


def lines(txt: str, comment: str = None) -> t.List[str]:
    """Convert a multiline string into a list of strings.

    Strip each line, skip empty lines, if `comment` is given, also remove lines starting with it.
    """

    ls = []

    for s in txt.splitlines():
        if comment:
            s = s.split(comment)[0]
        s = s.strip()
        if s:
            ls.append(s)

    return ls


def read_file(path, mode='rt'):
    with open(path, mode) as fp:
        return fp.read()


def write_file(path, s, mode='wt'):
    with open(path, mode) as fp:
        return fp.write(s)


def ensure_dir(dir_path: str, base_dir: str = None, mode: int = 0o755, user: int = None, group: int = None) -> str:
    """Check if a (possibly nested) directory exists and create if it does not.

    Args:
        dir_path: Path to a directory.
        base_dir: Base directory.
        mode: Directory creation mode.
        user: Directory user (defaults to gws.UID)
        group: Directory group (defaults to gws.GID)

    Retruns:
        The absolute path to the directory.
    """

    if base_dir:
        if os.path.isabs(dir_path):
            raise ValueError(f'cannot use an absolute path {dir_path!r} with a base dir')
        bpath = os.path.join(base_dir.encode('utf8'), dir_path.encode('utf8'))
    else:
        if not os.path.isabs(dir_path):
            raise ValueError(f'cannot use a relative path {dir_path!r} without a base dir')
        bpath = dir_path.encode('utf8')

    parts = []

    for p in bpath.split(b'/'):
        parts.append(p)
        path = b'/'.join(parts)
        if path and not os.path.isdir(path):
            os.mkdir(path, mode)

    os.chown(bpath, user or gws.core.const.UID, group or gws.core.const.GID)
    return bpath.decode('utf8')


def random_string(size: int) -> str:
    """Generate a random string of length `size`. """

    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    r = random.SystemRandom()
    return ''.join(r.choice(a) for _ in range(size))


def sha256(s):
    return hashlib.sha256(as_bytes(s)).hexdigest()


class cached_property:
    """Decorator for a cached property."""

    def __init__(self, fn):
        self._fn = fn
        self.__doc__ = getattr(fn, '__doc__')

    def __get__(self, obj, objtype=None):
        value = self._fn(obj)
        setattr(obj, self._fn.__name__, value)
        return value


_global_lock = threading.RLock()
_global_vars = {}


def global_lock():
    return _global_lock


def get_global(name, init_fn):
    """Get a global variable in a thread-safe way.

    Args:
        name: Variable name
        init_fn: Function that returns the value if the name doesn't exist.

    Returns:
        The variable value
    """

    global _global_vars

    if name in _global_vars:
        return _global_vars[name]

    with global_lock():
        if name in _global_vars:
            return _global_vars[name]
        _global_vars[name] = init_fn()

    return _global_vars[name]


def set_global(name, value):
    """Set a global variable in a thread-safe way.

    Args:
        name: Variable name.
        value: Variable value.

    Returns:
        The value
    """

    global _global_vars

    with global_lock():
        _global_vars[name] = value

    return _global_vars[name]


def get_cached_object(name, init_fn, max_age: int):
    """Return a cached object pickled in gws.OBJECT_CACHE_DIR.

    Args:
        name: Object name
        init_fn: Function that returns the value if the cache doesn't exist or is too old
        max_age: Cache max age in seconds.

    Returns:
         The value.
    """

    path = gws.core.const.OBJECT_CACHE_DIR + '/' + as_uid(name)

    with global_lock():

        try:
            age = time.time() - os.stat(path).st_mtime
        except:
            age = -1

        if 0 <= age < max_age:
            try:
                with open(path, 'rb') as fp:
                    return pickle.load(fp)
            except:
                pass

        try:
            os.unlink(path)
        except:
            pass

        obj = init_fn()

        if obj:
            with open(path, 'wb') as fp:
                pickle.dump(obj, fp)

        return obj


def running_in_container() -> bool:
    """True if the app is running in a docker container.

    Check the marker file created by our docker build (see install/build.py)
    """

    try:
        return os.path.isfile('/.GWS_IN_CONTAINER')
    except:
        return False


####################################################################################################


def _get(x, keys):
    for k in keys:
        if isinstance(x, dict):
            x = x[k]
        elif _is_list(x):
            x = x[int(k)]
        else:
            x = getattr(x, k)
    return x


def _is_list(x):
    return isinstance(x, (tuple, list))


def _is_bytes(x):
    return isinstance(x, (bytes, bytearray))
    # @TODO how to handle bytes-alikes?
    # return hasattr(x, 'decode')


def _is_dict(x):
    return isinstance(x, dict)


_QS_SAFE = b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-'

_QS_MAP = {
    c: bytes([c]) if c in _QS_SAFE else b'%%%02X' % c
    for c in range(256)
}


def _qs_quote(x):
    return b''.join(_QS_MAP[c] for c in _qs_bytes(x))


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
