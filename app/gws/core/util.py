"""Core utilities

Most common function which are needed everywhere. These function are exported in `gws` and can be used as gws.function().
"""

import hashlib
import importlib
import json
import os
import pickle
import random
import re
import sys
import threading
import time
import urllib.parse

import gws

from . import const, log
from .data import Data, is_data_object
from gws.types import List, cast


def exit(code: int = 255):
    """Exit the application.

    Args:
        code: Exit code.
    """

    sys.exit(code)


##

# @TODO use ABC

def is_list(x):
    return isinstance(x, (list, tuple))


def is_dict(x):
    return isinstance(x, dict)


def is_bytes(x):
    return isinstance(x, (bytes, bytearray))
    # @TODO how to handle bytes-alikes?
    # return hasattr(x, 'decode')


def is_atom(x):
    return x is None or isinstance(x, (int, float, bool, str, bytes))


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


##

def get(x, key, default=None):
    """Get a nested value/attribute from a structure.

    Args:
        x: A dict, list or Data.
        key: A list or a dot separated string of nested keys.
        default: The default value.

    Returns:
        The value if it exists and the default otherwise.
    """

    if not x:
        return default
    if isinstance(key, str):
        key = key.split('.')
    try:
        return _get(x, key)
    except (KeyError, IndexError, AttributeError, ValueError):
        return default


def has(x, key) -> bool:
    """True if a nested value/attribute exists in a structure.

    Args:
        x: A dict, list or Data.
        key: A list or a dot separated string of nested keys.

    Returns:
        True if a key exists
    """

    if not x:
        return False
    if isinstance(key, str):
        key = key.split('.')
    try:
        _get(x, key)
        return True
    except (KeyError, IndexError, AttributeError, ValueError):
        return False


def _get(x, keys):
    for k in keys:
        if is_dict(x):
            x = x[k]
        elif is_list(x):
            x = x[int(k)]
        elif is_data_object(x):
            v = getattr(x, k)
            if v is None:
                # special case: raise a KeyError if the attribute is truly missing in a Data
                # (and not just equals to None)
                v = vars(x)[k]
            x = v
        else:
            x = getattr(x, k)
    return x


def pop(x, key, default=None):
    if is_dict(x):
        return x.pop(key, default)
    if is_data_object(x):
        return vars(x).pop(key, default)
    return default


def pick(x, *keys):
    def _pick(d, keys):
        r = {}
        for k in keys:
            if k in d:
                r[k] = d[k]
        return r

    if is_dict(x):
        return _pick(x, keys)
    if is_data_object(x):
        return type(x)(_pick(vars(x), keys))
    return {}


def merge(x, *args, **kwargs):
    """Create a new dict/Data object by merging values from dicts/Datas or kwargs.
    Latter vales overwrite former ones unless None.

    Args:
        x: dict or Data.
        *args: dicts or Datas.
        **kwargs: Keyword args.

    Returns:
        A new object (dict or Data).
    """

    d = {}

    for a in [x, *args, kwargs]:
        for k, v in to_dict(a).items():
            if v is not None:
                d[k] = v

    return d if is_dict(x) else type(x)(d)


def deep_merge(x, y, concat_lists=True):
    """Deeply merge dicts/Datas into a nested dict/Data.
    Latter vales overwrite former ones unless None.

    Args:
        *x: dict or Data.
        *y: dict or Data.
        *concat_lists: if true, list will be concatenated, otherwise merged

    Returns:
        A new object (dict or Data).
    """

    def _merge(x1, y1):
        if (is_dict(x1) or is_data_object(x1)) and (is_dict(y1) or is_data_object(y1)):
            return deep_merge(x1, y1, concat_lists)

        if is_list(x1) and is_list(y1):
            x1 = compact(x1)
            y1 = compact(y1)
            if concat_lists:
                return x1 + y1
            return [_merge(x2, y2) for x2, y2 in zip(x1, y1)]

        return y1 if y1 is not None else x1

    xd = to_dict(x)
    yd = to_dict(y)

    d = {
        k: _merge(xd.get(k), yd.get(k))
        for k in xd.keys() | yd.keys()
    }

    return d if is_dict(x) else type(x)(d)


def compact(x):
    """Remove all None values from a collection."""

    if is_dict(x):
        return {k: v for k, v in x.items() if v is not None}
    if is_data_object(x):
        d = {k: v for k, v in vars(x).items() if v is not None}
        return type(x)(d)
    return [v for v in x if v is not None]


def strip(x):
    """Strip all strings and remove empty values from a collection."""

    def _strip(v):
        if isinstance(v, (str, bytes, bytearray)):
            return v.strip()
        return v

    def _dict(x1):
        d = {}
        for k, v in x1.items():
            v = _strip(v)
            if not is_empty(v):
                d[k] = v
        return d

    if is_dict(x):
        return _dict(x)
    if is_data_object(x):
        return type(x)(_dict(vars(x)))

    r = [_strip(v) for v in x]
    return [v for v in r if not is_empty(v)]


def uniq(x):
    """Remove duplicate elements from a collection."""

    s = set()
    r = []

    for y in x:
        try:
            if y not in s:
                s.add(y)
                r.append(y)
        except TypeError:
            if y not in r:
                r.append(y)

    return r


##

def to_int(x) -> int:
    """Convert a value to an int or 0 if this fails."""

    try:
        return int(x)
    except:
        return 0


def to_float(x) -> float:
    """Convert a value to a float or 0.0 if this fails."""

    try:
        return float(x)
    except:
        return 0.0


def to_str(x, encodings: List[str] = None) -> str:
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
    if not is_bytes(x):
        return str(x)
    if encodings:
        for enc in encodings:
            try:
                return x.decode(encoding=enc, errors='strict')
            except UnicodeDecodeError:
                pass
    return x.decode(encoding='utf-8', errors='ignore')


def to_bytes(x) -> bytes:
    """Convert a value to bytes by converting it to string and encoding in utf8."""

    if is_bytes(x):
        return bytes(x)
    if not isinstance(x, str):
        x = str(x)
    return x.encode('utf8')


def to_list(x, delimiter: str = ',') -> list:
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
    if is_bytes(x):
        x = to_str(x)
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


def to_dict(x) -> dict:
    """Convert a value to a dict. If the argument is a Data object, return its `dict`."""

    if is_dict(x):
        return x
    if is_data_object(x):
        return vars(x)
    return {}


def to_upper_dict(x) -> dict:
    x = to_dict(x)
    return {k.upper(): v for k, v in x.items()}


def to_lower_dict(x) -> dict:
    x = to_dict(x)
    return {k.lower(): v for k, v in x.items()}


def to_data(x) -> Data:
    """Convert a value to a Data. If the argument is a Data object, return it."""

    if is_data_object(x):
        return x
    if is_dict(x):
        return gws.Data(x)
    return gws.Data()


##

_UID_DE_TRANS = {
    ord('ä'): 'ae',
    ord('ö'): 'oe',
    ord('ü'): 'ue',
    ord('ß'): 'ss',
}


def to_uid(x) -> str:
    """Convert a value to an uid (alphanumeric string)."""

    if not x:
        return ''
    x = to_str(x).lower().strip().translate(_UID_DE_TRANS)
    x = re.sub(r'[^a-z0-9]+', '_', x)
    return x.strip('_')


def to_lines(txt: str, comment: str = None) -> List[str]:
    """Convert a multiline string into a list of strings.

    Strip each line, skip empty lines, if `comment` is given, also remove lines starting with it.
    """

    ls = []

    for s in txt.splitlines():
        if comment and comment in s:
            s = s.split(comment)[0]
        s = s.strip()
        if s:
            ls.append(s)

    return ls


##


def is_file(path):
    return os.path.isfile(path)


def is_dir(path):
    return os.path.isdir(path)


def read_file(path: str) -> str:
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


def read_file_b(path: str) -> bytes:
    with open(path, 'rb') as fp:
        return fp.read()


def write_file(path: str, s: str, user: int = None, group: int = None):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(s)
    _chown(path, user, group)


def write_file_b(path: str, s: bytes, user: int = None, group: int = None):
    with open(path, 'wb') as fp:
        fp.write(s)
    _chown(path, user, group)


def dirname(path):
    return os.path.dirname(path)


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
        bpath = cast(bytes, os.path.join(base_dir.encode('utf8'), dir_path.encode('utf8')))
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

    _chown(bpath, user, group)
    return bpath.decode('utf8')


def ensure_system_dirs():
    ensure_dir(gws.CONFIG_DIR)
    ensure_dir(gws.GLOBALS_DIR)
    ensure_dir(gws.LEGEND_CACHE_DIR)
    ensure_dir(gws.LOCKS_DIR)
    ensure_dir(gws.LOG_DIR)
    ensure_dir(gws.MAPPROXY_CACHE_DIR)
    ensure_dir(gws.MISC_DIR)
    ensure_dir(gws.NET_CACHE_DIR)
    ensure_dir(gws.OBJECT_CACHE_DIR)
    ensure_dir(gws.PRINT_DIR)
    ensure_dir(gws.SERVER_DIR)
    ensure_dir(gws.SPOOL_DIR)
    ensure_dir(gws.WEB_CACHE_DIR)


def _chown(path, user, group):
    try:
        os.chown(path, user or const.UID, group or const.GID)
    except OSError:
        pass


def random_string(size: int) -> str:
    """Generate a random string of length `size`. """

    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    r = random.SystemRandom()
    return ''.join(r.choice(a) for _ in range(size))


def sha256(x):
    def _bytes(x):
        if is_bytes(x):
            return bytes(x)
        if isinstance(x, (int, float, bool)):
            return str(x).encode('utf8')
        if isinstance(x, str):
            return x.encode('utf8')

    def _default(x):
        if gws.is_data_object(x):
            return vars(x)
        return str(x)

    c = _bytes(x)
    if c is None:
        j = json.dumps(x, default=_default, sort_keys=True, ensure_ascii=True)
        c = j.encode('utf8')

    return hashlib.sha256(c).hexdigest()


class cached_property:
    """Decorator for a cached property."""

    def __init__(self, fn):
        self._fn = fn
        self.__doc__ = getattr(fn, '__doc__')

    def __get__(self, obj, objtype=None):
        value = self._fn(obj)
        setattr(obj, self._fn.__name__, value)
        return value


# application lock/globals are global to one application
# server locks lock the whole server
# server globals are pickled in /tmp


_app_lock = threading.RLock()


def app_lock(name=''):
    return _app_lock


_app_globals: dict = {}


def get_app_global(name, init_fn):
    if name in _app_globals:
        return _app_globals[name]

    with app_lock(name):
        if name not in _app_globals:
            _app_globals[name] = init_fn()

    return _app_globals[name]


def set_app_global(name, value):
    with app_lock(name):
        _app_globals[name] = value
    return _app_globals[name]


def delete_app_global(name):
    with app_lock(name):
        _app_globals.pop(name, None)


##

def serialize_to_path(obj, path):
    tmp = path + random_string(64)
    with open(tmp, 'wb') as fp:
        pickle.dump(obj, fp)
    os.replace(tmp, path)


def unserialize_from_path(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


_server_globals = {}


def get_server_global(name, init_fn):
    uid = to_uid(name)
    path = gws.GLOBALS_DIR + '/' + uid

    def _get():
        if uid in _server_globals:
            log.debug(f'get_server_global {uid!r} - found')
            return True

        if os.path.isfile(path):
            try:
                _server_globals[uid] = unserialize_from_path(path)
                log.debug(f'get_server_global {uid!r} - loaded')
                return True
            except:
                log.exception(f'get_server_global {uid!r} LOAD ERROR')

    if _get():
        return _server_globals[uid]

    with server_lock(uid):

        if _get():
            return _server_globals[uid]

        _server_globals[uid] = init_fn()

        try:
            serialize_to_path(_server_globals[uid], path)
            log.debug(f'get_server_global {uid!r} - stored')
        except:
            log.exception(f'get_server_global {uid!r} STORE ERROR')

        return _server_globals[uid]


class _FileLock:
    _PAUSE = 2
    _TIMEOUT = 60

    def __init__(self, uid):
        self.uid = to_uid(uid)
        self.path = gws.LOCKS_DIR + '/' + self.uid

    def __enter__(self):
        self.acquire()
        log.debug(f'server lock {self.uid!r} ACQUIRED')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def acquire(self):
        ts = time.time()

        while True:
            try:
                fp = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fp, bytes(os.getpid()))
                os.close(fp)
                return
            except:
                pass

            t = time.time() - ts

            if t > self._TIMEOUT:
                raise ValueError('lock timeout', self.uid)

            log.debug(f'server lock {self.uid!r} WAITING time={t:.3f}')
            time.sleep(self._PAUSE)

    def release(self):
        try:
            os.unlink(self.path)
            log.debug(f'server lock {self.uid!r} RELEASED')
        except:
            log.exception(f'server lock {self.uid!r} RELEASE ERROR')


def server_lock(uid):
    return _FileLock(uid)


##

def import_from_path(path):
    in_path, root, mod = _find_import_root_and_module_name(path)
    gws.log.debug(f'import_from_path: in_path={in_path} path={path!r} root={root!r} mod={mod!r}')
    if not in_path:
        sys.path.insert(0, root)
    return importlib.import_module(mod)


def _find_import_root_and_module_name(path):
    if not os.path.isabs(path):
        path = gws.APP_DIR + '/' + path

    init = '__init__.py'
    path = os.path.normpath(path)

    if os.path.isdir(path):
        path += '/' + init
    if not os.path.isfile(path):
        raise ValueError(f'import_from_path: {path!r}: not found')

    dirname, filename = os.path.split(path)
    lastmod = [] if filename == init else [filename.split('.')[0]]
    dirs = dirname.strip('/').split('/')

    # first, try to locate the longest directory root in sys.path

    for n in range(len(dirs), 0, -1):
        root = '/' + '/'.join(dirs[:n])
        if root in sys.path:
            return True, root, '.'.join(dirs[n:] + lastmod)

    # second, find the longest path that doesn't contain __init__.py
    # this will be a new root

    for n in range(len(dirs), 0, -1):
        root = '/' + '/'.join(dirs[:n])
        if not os.path.isfile(root + '/' + init):
            return False, root, '.'.join(dirs[n:] + lastmod)

    raise ValueError(f'import_from_path: {path!r}: cannot be imported')


##

def action_url_path(name: str, **kwargs) -> str:
    ls = []

    for k, v in kwargs.items():
        if not is_empty(v):
            ls.append(urllib.parse.quote(k))
            ls.append(urllib.parse.quote(to_str(v)))

    path = gws.SERVER_ENDPOINT + '/' + name
    if ls:
        path += '/' + '/'.join(ls)
    return path
