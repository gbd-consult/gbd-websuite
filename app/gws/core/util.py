"""Core utilities

Most common function which are needed everywhere.

This module is available as ``gws.u`` everywhere.
"""

import hashlib
import json
import os
import shutil
import pickle
import random
import re
import sys
import threading
import time
import urllib.parse

from typing import cast, Union

from . import const, log


def is_data_object(x) -> bool:
    pass


def to_data_object(x) -> bool:
    pass


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
            # special case: raise a KeyError if the attribute is truly missing in a Data
            # (and not just equals to None)
            x = vars(x)[k]
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
    def _pick(d):
        r = {}
        for k in keys:
            if k in d:
                r[k] = d[k]
        return r

    if is_dict(x):
        return _pick(x)
    if is_data_object(x):
        return type(x)(_pick(vars(x)))
    return {}


def omit(x, *keys):
    def _omit(d):
        r = {}
        for k, v in d.items():
            if k not in keys:
                r[k] = d[k]
        return r

    if is_dict(x):
        return _omit(x)
    if is_data_object(x):
        return type(x)(_omit(vars(x)))
    return {}


def collect(pairs):
    m = {}

    for key, val in pairs:
        if key is not None:
            m.setdefault(key, []).append(val)

    return m


def first(it):
    for x in it:
        return x


def first_not_none(*args):
    for a in args:
        if a is not None:
            return a


def merge(*args, **kwargs) -> Union[dict, 'Data']:
    """Create a new dict/Data object by merging values from dicts/Datas or kwargs.
    Latter vales overwrite former ones unless None.

    Args:
        *args: dicts or Datas.
        **kwargs: Keyword args.

    Returns:
        A new object (dict or Data).
    """

    def _merge(arg):
        for k, v in to_dict(arg).items():
            if v is not None:
                m[k] = v

    m = {}

    for a in args:
        if a:
            _merge(a)
    if kwargs:
        _merge(kwargs)

    if not args or isinstance(args[0], dict) or args[0] is None:
        return m
    return type(args[0])(m)


def deep_merge(x, y, concat_lists=True):
    """Deeply merge dicts/Datas into a nested dict/Data.
    Latter vales overwrite former ones unless None.

    Args:
        x: dict or Data.
        y: dict or Data.
        concat_lists: if true, list will be concatenated, otherwise merged

    Returns:
        A new object (dict or Data).
    """

    if (is_dict(x) or is_data_object(x)) and (is_dict(y) or is_data_object(y)):
        xd = to_dict(x)
        yd = to_dict(y)
        d = {
            k: deep_merge(xd.get(k), yd.get(k), concat_lists)
            for k in xd.keys() | yd.keys()
        }
        return d if is_dict(x) else type(x)(d)

    if is_list(x) and is_list(y):
        xc = compact(x)
        yc = compact(y)
        if concat_lists:
            return xc + yc
        return [deep_merge(x1, y1, concat_lists) for x1, y1 in zip(xc, yc)]

    return y if y is not None else x


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


def to_rounded_int(x) -> int:
    """Round and convert a value to an int or 0 if this fails."""

    try:
        if isinstance(x, float):
            return int(round(x))
        return int(x)
    except:
        return 0


def to_float(x) -> float:
    """Convert a value to a float or 0.0 if this fails."""

    try:
        return float(x)
    except:
        return 0.0


def to_str(x, encodings: list[str] = None) -> str:
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
    if x is None:
        return ''
    if not is_bytes(x):
        return str(x)
    if encodings:
        for enc in encodings:
            try:
                return x.decode(encoding=enc, errors='strict')
            except UnicodeDecodeError:
                pass
    return x.decode(encoding='utf-8', errors='ignore')


def to_bytes(x, encoding='utf8') -> bytes:
    """Convert a value to bytes by converting it to string and encoding."""

    if is_bytes(x):
        return bytes(x)
    if x is None:
        return b''
    if not isinstance(x, str):
        x = str(x)
    return x.encode(encoding or 'utf8')


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
    """Convert a value to a dict. If the argument is an object, return its `dict`."""

    if is_dict(x):
        return x
    if x is None:
        return {}
    try:
        f = getattr(x, '_asdict', None)
        if f:
            return f()
        return vars(x)
    except TypeError:
        raise ValueError(f'cannot convert {x!r} to dict')


def to_upper_dict(x) -> dict:
    x = to_dict(x)
    return {k.upper(): v for k, v in x.items()}


def to_lower_dict(x) -> dict:
    x = to_dict(x)
    return {k.lower(): v for k, v in x.items()}


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


def to_lines(txt: str, comment: str = None) -> list[str]:
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

def parse_acl(acl):
    """Parse an ACL config into an ACL.

    Args:
        acl: an ACL config. Can be given as a string ``allow X, allow Y, deny Z``,
            or as a list of dicts ``{ role X type allow }, { role Y type deny }``,
            or it can already be an ACL ``[1 X], [0 Y]``,
            or it can be None.

    Returns:
        Access list.
    """

    if not acl:
        return []

    a = 'allow'
    d = 'deny'
    bits = {const.ALLOW, const.DENY}
    err = 'invalid ACL'

    access = []

    if isinstance(acl, str):
        for p in acl.strip().split(','):
            s = p.strip().split()
            if len(s) != 2:
                raise ValueError(err)
            if s[0] == a:
                access.append((const.ALLOW, s[1]))
            elif s[0] == d:
                access.append((const.DENY, s[1]))
            else:
                raise ValueError(err)
        return access

    if not isinstance(acl, list):
        raise ValueError(err)

    if isinstance(acl[0], (list, tuple)):
        try:
            if all(len(s) == 2 and s[0] in bits for s in acl):
                return acl
        except (TypeError, IndexError):
            pass
        raise ValueError(err)

    if isinstance(acl[0], dict):
        for s in acl:
            tk = s.get('type', '')
            rk = s.get('role', '')
            if not isinstance(rk, str):
                raise ValueError(err)
            if tk == a:
                access.append((const.ALLOW, rk))
            elif tk == d:
                access.append((const.DENY, rk))
            else:
                raise ValueError(err)
        return access

    raise ValueError(err)


##

UID_DELIMITER = '::'


def join_uid(parent_uid, object_uid):
    p = parent_uid.split(UID_DELIMITER)
    u = object_uid.split(UID_DELIMITER)
    return p[-1] + UID_DELIMITER + u[-1]


def split_uid(joined_uid: str) -> tuple[str, str]:
    p, _, u = joined_uid.partition(UID_DELIMITER)
    return p, u


##

def is_file(path):
    return os.path.isfile(path)


def is_dir(path):
    return os.path.isdir(path)


def read_file(path: str) -> str:
    try:
        with open(path, 'rt', encoding='utf8') as fp:
            return fp.read()
    except Exception as exc:
        log.debug(f'error reading {path=} {exc=}')
        raise


def read_file_b(path: str) -> bytes:
    try:
        with open(path, 'rb') as fp:
            return fp.read()
    except Exception as exc:
        log.debug(f'error reading {path=} {exc=}')
        raise


def write_file(path: str, s: str, user: int = None, group: int = None):
    try:
        with open(path, 'wt', encoding='utf8') as fp:
            fp.write(s)
        chown_default(path, user, group)
        return path
    except Exception as exc:
        log.debug(f'error writing {path=} {exc=}')
        raise


def write_file_b(path: str, s: bytes, user: int = None, group: int = None):
    try:
        with open(path, 'wb') as fp:
            fp.write(s)
        chown_default(path, user, group)
        return path
    except Exception as exc:
        log.debug(f'error writing {path=} {exc=}')
        raise


def dirname(path):
    return os.path.dirname(path)


def ensure_dir(dir_path: str, base_dir: str = None, mode: int = 0o755, user: int = None, group: int = None) -> str:
    """Check if a (possibly nested) directory exists and create if it does not.

    Args:
        dir_path: Path to a directory.
        base_dir: Base directory.
        mode: Directory creation mode.
        user: Directory user (defaults to gws.c.UID)
        group: Directory group (defaults to gws.c.GID)

    Returns:
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

    chown_default(bpath, user, group)
    return bpath.decode('utf8')


def ensure_system_dirs():
    for d in const.ALL_DIRS:
        ensure_dir(d)


def chown_default(path, user=None, group=None):
    try:
        os.chown(path, user or const.UID, group or const.GID)
    except OSError:
        pass


_ephemeral_state = dict(
    last_check_time=0,
    check_interval=20 * 30,
    max_age=20 * 30,
)


def ephemeral_path(name: str) -> str:
    """Return a new ephemeral path name."""

    # if stime() > _ephemeral_state['last_check_time'] + _ephemeral_state['check_interval']:
    #     _ephemeral_cleanup()
    #
    name = str(os.getpid()) + '_' + random_string(64) + '_' + name
    return const.EPHEMERAL_DIR + '/' + name


def ephemeral_dir(name: str) -> str:
    """Create and return an ephemeral directory."""

    return ensure_dir(const.EPHEMERAL_DIR + '/' + name)


def ephemeral_cleanup():
    """Remove ephemeral paths older than max age."""

    cnt = 0
    ts = stime()

    for de in os.scandir(const.EPHEMERAL_DIR):
        age = int(ts - de.stat().st_mtime)
        if age > _ephemeral_state['max_age']:
            try:
                if de.is_dir():
                    shutil.rmtree(de.path)
                else:
                    os.unlink(de.path)
                cnt += 1
            except OSError:
                pass

    _ephemeral_state['last_check_time'] = ts

    if cnt > 0:
        log.debug(f'_ephemeral_cleanup: {cnt}')


def random_string(size: int) -> str:
    """Generate a random string of length `size`. """

    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    r = random.SystemRandom()
    return ''.join(r.choice(a) for _ in range(size))


class _FormatMapDefault:
    def __init__(self, d, default):
        self.d = d
        self.default = default

    def __getitem__(self, item):
        val = self.d.get(item)
        return val if val is not None else self.default


def format_map(fmt: str, x: Union[dict, 'Data'], default: str = '') -> str:
    return fmt.format_map(_FormatMapDefault(x, default))


def sha256(x):
    def _bytes(x):
        if is_bytes(x):
            return bytes(x)
        if isinstance(x, (int, float, bool)):
            return str(x).encode('utf8')
        if isinstance(x, str):
            return x.encode('utf8')

    def _default(x):
        if is_data_object(x):
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
    chown_default(path)
    return path


def unserialize_from_path(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


_server_globals = {}


def get_cached_object(name: str, life_time: int, init_fn):
    uid = to_uid(name)
    path = const.OBJECT_CACHE_DIR + '/' + uid

    def _get():
        if not os.path.isfile(path):
            return
        try:
            age = int(time.time() - os.stat(path).st_mtime)
        except OSError:
            return
        if age < life_time:
            try:
                obj = unserialize_from_path(path)
                log.debug(f'get_cached_object {uid!r} {life_time=} {age=} - loaded')
                return obj
            except:
                log.exception(f'get_cached_object {uid!r} LOAD ERROR')

    obj = _get()
    if obj:
        return obj

    with server_lock(uid):
        obj = _get()
        if obj:
            return obj

        obj = init_fn()
        try:
            serialize_to_path(obj, path)
            log.debug(f'get_cached_object {uid!r} - stored')
        except:
            log.exception(f'get_cached_object {uid!r} STORE ERROR')

        return obj


def get_server_global(name: str, init_fn):
    uid = to_uid(name)
    path = const.GLOBALS_DIR + '/' + uid

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
        self.path = const.LOCKS_DIR + '/' + self.uid

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
            except FileExistsError:
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

def action_url_path(name: str, **kwargs) -> str:
    ls = []

    for k, v in kwargs.items():
        if not is_empty(v):
            ls.append(urllib.parse.quote(k))
            ls.append(urllib.parse.quote(to_str(v)))

    path = const.SERVER_ENDPOINT + '/' + name
    if ls:
        path += '/' + '/'.join(ls)
    return path


##

def utime() -> float:
    """Unix time as a float number."""
    return time.time()


def stime() -> int:
    """Unix time as an integer number of seconds."""
    return int(time.time())


def sleep(n: float):
    """Sleep for n seconds."""
    time.sleep(n)


def mstime() -> int:
    """Unix time as an integer number of milliseconds."""
    return int(time.time() * 1000)


def microtime() -> int:
    """Unix time as an integer number of microseconds."""
    return int(time.time() * 1000000)
