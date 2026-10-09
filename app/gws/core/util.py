"""General utilities, available as ``gws.u``."""

import fcntl
import hashlib
import json
import os
import pickle
import random
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from typing import Optional, TypeVar, Union, cast

from . import const, log


def is_data_object(x) -> bool:
    """Check if the argument is a ``Data`` object.

    This is a placeholder, replaced by ``gws.is_data_object`` when ``gws`` is imported.

    Args:
        x: A value.

    Returns:
        ``True`` if the value is a ``Data`` object.
    """
    return False


def to_data_object(x):
    """Convert a value to a ``Data`` object.

    This is a placeholder, replaced by ``gws.to_data_object`` when ``gws`` is imported.

    Args:
        x: A value.

    Returns:
        A ``Data`` object.
    """
    pass


def exit(code: int = 255):
    """Exit the application.

    Args:
        code: Exit code.
    """

    sys.exit(code)


T = TypeVar('T')


def require(value: Optional[T], message: str = '') -> T:
    """Return the value if it is not ``None``, otherwise raise an error.

    Args:
        value: A value.
        message: Error message.

    Returns:
        The value.

    Raises:
        ValueError: If the value is ``None``.
    """
    if value is None:
        raise ValueError(message or 'unexpected None value')
    return value


##

# @TODO use ABC


def is_list(x):
    """Check if the value is a list or a tuple.

    Args:
        x: A value.

    Returns:
        ``True`` if the value is a list or a tuple.
    """
    return isinstance(x, (list, tuple))


def is_dict(x):
    """Check if the value is a dict.

    Args:
        x: A value.

    Returns:
        ``True`` if the value is a dict.
    """
    return isinstance(x, dict)


def is_bytes(x):
    """Check if the value is ``bytes`` or ``bytearray``.

    Args:
        x: A value.

    Returns:
        ``True`` if the value is ``bytes`` or ``bytearray``.
    """
    return isinstance(x, (bytes, bytearray))
    # @TODO how to handle bytes-alikes?
    # return hasattr(x, 'decode')


def is_atom(x):
    """Check if the value is ``None`` or a scalar (number, bool, string or bytes).

    Args:
        x: A value.

    Returns:
        ``True`` if the value is an atom.
    """
    return x is None or isinstance(x, (int, float, bool, str, bytes))


def is_empty(x) -> bool:
    """Check if the value is empty.

    A value is empty if it is ``None``, has zero length, or is an object without attributes.

    Args:
        x: A value.

    Returns:
        ``True`` if the value is empty.
    """

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
    """Get a nested value or attribute from a structure.

    Args:
        x: A dict, list, ``Data`` or any object.
        key: A list or a dot-separated string of nested keys. List elements are addressed by numeric keys.
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
    """Check if a nested value or attribute exists in a structure.

    Args:
        x: A dict, list, ``Data`` or any object.
        key: A list or a dot-separated string of nested keys.

    Returns:
        ``True`` if the key exists.
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
    """Follow the keys into a structure, raise an error if a key is missing."""
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
    """Remove a key from a dict or a ``Data`` object and return its value.

    Args:
        x: A dict or ``Data``.
        key: The key.
        default: Value to return if the key is missing or ``x`` is of another type.

    Returns:
        The value or the default.
    """
    if is_dict(x):
        return x.pop(key, default)
    if is_data_object(x):
        return vars(x).pop(key, default)
    return default


def pick(x, *keys):
    """Return a copy of a dict or a ``Data`` object with the given keys only.

    Args:
        x: A dict or ``Data``.
        *keys: Keys to keep.

    Returns:
        A new object of the same type, or an empty dict if ``x`` is of another type.
    """
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
    """Return a copy of a dict or a ``Data`` object without the given keys.

    Args:
        x: A dict or ``Data``.
        *keys: Keys to remove.

    Returns:
        A new object of the same type, or an empty dict if ``x`` is of another type.
    """
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
    """Group values by keys.

    Args:
        pairs: An iterable of ``(key, value)`` pairs. Pairs with a ``None`` key are skipped.

    Returns:
        A dict mapping each key to the list of its values.
    """
    m = {}

    for key, val in pairs:
        if key is not None:
            m.setdefault(key, []).append(val)

    return m


def first(it):
    """Return the first element of an iterable.

    Args:
        it: An iterable.

    Returns:
        The first element, or ``None`` if the iterable is empty.
    """
    for x in it:
        return x


def first_not_none(*args):
    """Return the first argument that is not ``None``.

    Args:
        *args: Values.

    Returns:
        The first value that is not ``None``, or ``None``.
    """
    for a in args:
        if a is not None:
            return a


def merge(*args, **kwargs) -> Union[dict, 'Data']:
    """Create a new dict or ``Data`` object by merging values from dicts, ``Data`` objects and keyword args.

    Later values override earlier ones, unless they are ``None``.

    Args:
        *args: Dicts or ``Data`` objects. Empty values are skipped.
        **kwargs: Keyword args, merged last.

    Returns:
        A new object of the same type as the first argument, or a dict if it is a dict or ``None``.
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


def compact(x):
    """Remove all ``None`` values from a collection.

    Args:
        x: A dict, ``Data`` or an iterable.

    Returns:
        A new dict or ``Data`` object, or a list for other iterables.
    """

    if is_dict(x):
        return {k: v for k, v in x.items() if v is not None}
    if is_data_object(x):
        d = {k: v for k, v in vars(x).items() if v is not None}
        return type(x)(d)
    return [v for v in x if v is not None]


def strip(x):
    """Strip all strings and remove empty values from a collection.

    Args:
        x: A dict, ``Data`` or an iterable.

    Returns:
        A new dict or ``Data`` object, or a list for other iterables.
    """

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
    """Remove duplicate elements from a collection, keeping the order.

    Args:
        x: An iterable.

    Returns:
        A list of unique elements.
    """

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
    """Convert a value to an int.

    Args:
        x: A value.

    Returns:
        The int value, or 0 if the conversion fails.
    """

    try:
        return int(x)
    except:
        return 0


def to_rounded_int(x) -> int:
    """Round a float and convert a value to an int.

    Args:
        x: A value.

    Returns:
        The int value, or 0 if the conversion fails.
    """

    try:
        if isinstance(x, float):
            return int(round(x))
        return int(x)
    except:
        return 0


def to_float(x) -> float:
    """Convert a value to a float.

    Args:
        x: A value.

    Returns:
        The float value, or 0.0 if the conversion fails.
    """

    try:
        return float(x)
    except:
        return 0.0


def clamp(x, lo, hi):
    """Limit a value to a range.

    Args:
        x: A value.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        ``lo`` if the value is less than ``lo``, ``hi`` if it is greater than ``hi``, otherwise the value.
    """

    return max(lo, min(x, hi))


def to_str(x, encodings: list[str] = None) -> str:
    """Convert a value to a string.

    Bytes are decoded, ``None`` becomes an empty string, other values are converted with ``str``.

    Args:
        x: A value.
        encodings: A list of acceptable encodings. If the value is bytes, try each encoding
            and return the first result that decodes without errors. If none succeeds,
            decode as UTF-8 and ignore errors.

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
    """Convert a value to bytes by converting it to a string and encoding it.

    Args:
        x: A value. Bytes are returned as is, ``None`` becomes empty bytes.
        encoding: The encoding.

    Returns:
        Bytes.
    """

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
        x: A value. A string (or bytes) is split by the delimiter, the parts are stripped and empty parts removed.
            A number or a bool becomes a one-element list, other iterables are converted to a list.
        delimiter: The delimiter. If empty, a string becomes a one-element list.

    Returns:
        A list. Empty values and values that cannot be converted give an empty list.
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
    """Convert a value to a dict.

    Args:
        x: A dict, ``None``, a named tuple or an object.

    Returns:
        The dict itself, an empty dict for ``None``, the fields of a named tuple, or the object's ``vars``.

    Raises:
        ValueError: If the value cannot be converted.
    """

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


def to_json_value(x) -> Union[dict, list, str, int, float, bool, None]:
    """Recursively convert a value to a JSON serializable type.

    Args:
        x: A value. Dicts, lists and ``Data`` objects are converted recursively, other non-scalar values are converted with ``str``.

    Returns:
        A JSON serializable value.
    """

    if is_atom(x):
        return x
    if is_dict(x):
        return {k: to_json_value(v) for k, v in x.items()}
    if is_list(x):
        return [to_json_value(v) for v in x]
    if is_data_object(x):
        return {k: to_json_value(v) for k, v in vars(x).items()}
    return str(x)


def to_upper_dict(x) -> dict:
    """Convert a value to a dict with upper-case keys.

    Args:
        x: A value accepted by ``to_dict``.

    Returns:
        A new dict.
    """
    x = to_dict(x)
    return {k.upper(): v for k, v in x.items()}


def to_lower_dict(x) -> dict:
    """Convert a value to a dict with lower-case keys.

    Args:
        x: A value accepted by ``to_dict``.

    Returns:
        A new dict.
    """
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
    """Convert a value to a uid.

    The value is converted to a lower-case string, German umlauts are transliterated,
    and runs of other characters are replaced with underscores.

    Args:
        x: A value.

    Returns:
        A string of ``a-z``, ``0-9`` and ``_``, or an empty string for an empty value.
    """

    if not x:
        return ''
    x = to_str(x).lower().strip().translate(_UID_DE_TRANS)
    x = re.sub(r'[^a-z0-9]+', '_', x)
    return x.strip('_')


##


def parse_acl(acl):
    """Parse an ACL config into an ACL.

    Args:
        acl: An ACL config. Can be given as a string ``allow X, allow Y, deny Z``,
            or as a list of dicts ``{ role X type allow }, { role Y type deny }``,
            or it can already be an ACL ``[1 X], [0 Y]``,
            or it can be ``None``.

    Returns:
        Access list, empty for an empty value.

    Raises:
        ValueError: If the ACL config is invalid.
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
    """Join a parent uid and an object uid with ``UID_DELIMITER``.

    If either uid is already joined, only its last part is used.

    Args:
        parent_uid: Parent uid.
        object_uid: Object uid.

    Returns:
        The joined uid.
    """
    p = parent_uid.split(UID_DELIMITER)
    u = object_uid.split(UID_DELIMITER)
    return p[-1] + UID_DELIMITER + u[-1]


def split_uid(joined_uid: str) -> tuple[str, str]:
    """Split a joined uid at the first ``UID_DELIMITER``.

    Args:
        joined_uid: Joined uid.

    Returns:
        A tuple of the parent uid and the object uid. If there is no delimiter, the parent uid is empty.
    """
    p, sep, u = joined_uid.partition(UID_DELIMITER)
    if not sep:
        return '', joined_uid
    return p, u


##


def is_file(path):
    """Check if the path is an existing file.

    Args:
        path: File path.

    Returns:
        ``True`` if the file exists.
    """
    return os.path.isfile(path)


def is_dir(path):
    """Check if the path is an existing directory.

    Args:
        path: Directory path.

    Returns:
        ``True`` if the directory exists.
    """
    return os.path.isdir(path)


def read_file(path: str) -> str:
    """Read a UTF-8 text file.

    Args:
        path: File path.

    Returns:
        The file content.

    Raises:
        Exception: Errors from opening or reading the file are logged and raised again.
    """
    try:
        with open(path, 'rt', encoding='utf8') as fp:
            return fp.read()
    except Exception as exc:
        log.debug(f'error reading {path=} {exc=}')
        raise


def read_file_b(path: str) -> bytes:
    """Read a binary file.

    Args:
        path: File path.

    Returns:
        The file content.

    Raises:
        Exception: Errors from opening or reading the file are logged and raised again.
    """
    try:
        with open(path, 'rb') as fp:
            return fp.read()
    except Exception as exc:
        log.debug(f'error reading {path=} {exc=}')
        raise


def write_file(path: str, s: str, user: int = None, group: int = None):
    """Write a text file atomically, via a temporary file in the same directory.

    Args:
        path: File path.
        s: Text content, encoded as UTF-8.
        user: File owner, defaults to ``const.UID``.
        group: File group, defaults to ``const.GID``.

    Returns:
        The file path.

    Raises:
        Exception: Write errors are logged and raised again.
    """

    return write_file_b(path, s.encode('utf8'), user, group)


def write_file_b(path: str, s: str | bytes, user: int = None, group: int = None):
    """Write a binary file atomically, via a temporary file in the same directory.

    Args:
        path: File path.
        s: Content. A string is encoded as UTF-8.
        user: File owner, defaults to ``const.UID``.
        group: File group, defaults to ``const.GID``.

    Returns:
        The file path.

    Raises:
        Exception: Write errors are logged and raised again.
    """

    if isinstance(s, str):
        s = s.encode('utf8')
    tmp = f'{path}.{random_string(32)}.tmp'
    try:
        with open(tmp, 'wb') as fp:
            fp.write(s)
        chown_default(tmp, user, group)
        os.replace(tmp, path)
        return path
    except Exception as exc:
        log.debug(f'error writing {path=} {exc=}')
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_debug_file(path: str, s: str | bytes):
    """Write a file to the debug directory ``<VAR_DIR>/debug``.

    Errors are logged and ignored.

    Args:
        path: File path, relative to the debug directory.
        s: Content. A string is encoded as UTF-8.
    """

    if isinstance(s, str):
        s = s.encode('utf8')
    try:
        d = ensure_dir(f'{const.VAR_DIR}/debug')
        with open(f'{d}/{path}', 'wb') as fp:
            fp.write(s)
    except Exception as exc:
        log.debug(f'error writing debug {path=} {exc=}')


def dirname(path):
    """Return the directory part of a path.

    Args:
        path: A path.

    Returns:
        The directory name.
    """
    return os.path.dirname(path)


def ensure_dir(dir_path: str, base_dir: str = None, mode: int = 0o755, user: int = None, group: int = None) -> str:
    """Check if a (possibly nested) directory exists and create it if it does not.

    Args:
        dir_path: Path to a directory. Must be absolute without ``base_dir`` and relative with it.
        base_dir: Base directory.
        mode: Directory creation mode.
        user: Directory owner, defaults to ``const.UID``.
        group: Directory group, defaults to ``const.GID``.

    Returns:
        The path to the directory.

    Raises:
        ValueError: If the path is absolute with ``base_dir``, or relative without it.
    """

    if base_dir:
        if os.path.isabs(dir_path):
            raise ValueError(f'cannot use an absolute path {dir_path!r} with a base dir')
        bpath = cast(bytes, os.path.join(base_dir.encode('utf8'), dir_path.encode('utf8')))
    else:
        if not os.path.isabs(dir_path):
            raise ValueError(f'cannot use a relative path {dir_path!r} without a base dir')
        bpath = dir_path.encode('utf8')

    if os.path.isdir(bpath):
        return bpath.decode('utf8')

    parts = []

    for p in bpath.split(b'/'):
        parts.append(p)
        path = b'/'.join(parts)
        if path and not os.path.isdir(path):
            try:
                os.mkdir(path, mode)
                chown_default(path, user, group)
            except FileExistsError:
                pass

    return bpath.decode('utf8')


def ensure_system_dirs():
    """Create all system directories listed in ``const.ALL_DIRS``."""
    for d in const.ALL_DIRS:
        ensure_dir(d)


def chown_default(path, user=None, group=None):
    """Change the owner of a path, ignoring errors.

    Args:
        path: A path.
        user: Owner, defaults to ``const.UID``.
        group: Group, defaults to ``const.GID``.
    """
    try:
        os.chown(path, user or const.UID, group or const.GID)
    except OSError:
        pass


_ephemeral_state = dict(
    next_check=0,
    check_interval=2 * 3600,
    max_age=2 * 3600,
)


def ephemeral_path(name: str) -> str:
    """Return a new unique path in the ephemeral directory.

    The path is not created. Ephemeral paths are removed by ``ephemeral_cleanup`` after two hours.

    Args:
        name: Base name, appended to a unique prefix.

    Returns:
        The path.
    """

    ephemeral_cleanup()
    name = str(os.getpid()) + '_' + random_string(64) + '_' + name
    return const.EPHEMERAL_DIR + '/' + name


def ephemeral_dir(name: str) -> str:
    """Create a directory in the ephemeral directory, if it does not exist yet.

    Args:
        name: Directory name.

    Returns:
        The directory path.
    """

    ephemeral_cleanup()
    return ensure_dir(const.EPHEMERAL_DIR + '/' + name)


def ephemeral_cleanup(force=False):
    """Remove ephemeral files and empty directories older than two hours.

    Throttled to once per two hours in each process, unless ``force`` is set.

    Args:
        force: Run even if the last run was recent.

    Returns:
        The detached ``find`` process, or ``None`` if throttled.
    """

    ts = stime()

    if ts < _ephemeral_state['next_check'] and not force:
        return

    _ephemeral_state['next_check'] = ts + _ephemeral_state['check_interval']

    cutoff = ts - _ephemeral_state['max_age']
    cmd = [
        'find', const.EPHEMERAL_DIR, '-mindepth', '1',
        '!', '-newermt', f'@{cutoff}', '(', '-type', 'f', '-o', '-type', 'd', '-empty', ')',
        '-delete',
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def random_string(size: int) -> str:
    """Generate a random alphanumeric string.

    Args:
        size: String length.

    Returns:
        The string.
    """

    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    r = random.SystemRandom()
    return ''.join(r.choice(a) for _ in range(size))


class _FormatMapDefault:
    """Mapping for ``str.format_map`` that returns a default for missing or ``None`` values."""
    def __init__(self, d, default):
        self.d = d
        self.default = default

    def __getitem__(self, item):
        val = self.d.get(item)
        return val if val is not None else self.default


def format_map(fmt: str, x: Union[dict, 'Data'], default: str = '') -> str:
    """Format a string with values from a dict or a ``Data`` object.

    Args:
        fmt: Format string with ``{name}`` placeholders.
        x: A dict or ``Data`` object with the values.
        default: Replacement for missing or ``None`` values.

    Returns:
        The formatted string.
    """
    return fmt.format_map(_FormatMapDefault(x, default))


def sha256(x) -> str:
    """Compute the SHA-256 hash of a value.

    Bytes, strings and numbers are hashed directly, other values are converted to JSON first,
    with sorted keys and ``Data`` objects as dicts.

    Args:
        x: A value.

    Returns:
        The hex digest.
    """
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
    """Decorator for a cached property.

    The value is computed on first access and stored as an instance attribute with the same name.
    """

    def __init__(self, fn):
        """Create the descriptor.

        Args:
            fn: The getter function.
        """
        self._fn = fn
        self.__doc__ = getattr(fn, '__doc__')

    def __get__(self, obj, objtype=None):
        """Compute the value and store it on the instance."""
        value = self._fn(obj)
        setattr(obj, self._fn.__name__, value)
        return value


# application lock/globals are global to one application
# server locks lock the whole server
# server globals are pickled in /tmp


_app_locks: dict[str, threading.RLock] = {}
_app_master_lock = threading.Lock()


def app_lock(name=''):
    """Return a reentrant lock, local to this process.

    Locks with the same name are shared.

    Args:
        name: Lock name.

    Returns:
        A ``threading.RLock``.
    """
    with _app_master_lock:
        lock = _app_locks.get(name)
        if lock is None:
            lock = _app_locks[name] = threading.RLock()
        return lock


_app_globals: dict = {}


def get_app_global(name, init_fn):
    """Get a process-wide global value, creating it if needed.

    Args:
        name: Value name.
        init_fn: Function without arguments that creates the value. It is called once, under an app lock.

    Returns:
        The value.
    """
    if name in _app_globals:
        return _app_globals[name]

    with app_lock(name):
        if name not in _app_globals:
            _app_globals[name] = init_fn()

    return _app_globals[name]


def set_app_global(name, value):
    """Set a process-wide global value.

    Args:
        name: Value name.
        value: The value.

    Returns:
        The value.
    """
    with app_lock(name):
        _app_globals[name] = value
    return _app_globals[name]


def delete_app_global(name):
    """Delete a process-wide global value.

    Args:
        name: Value name.
    """
    with app_lock(name):
        _app_globals.pop(name, None)


##


def serialize_to_path(obj, path):
    """Pickle an object to a file atomically.

    Args:
        obj: An object.
        path: File path.

    Returns:
        The file path.
    """
    tmp = path + random_string(64)
    with open(tmp, 'wb') as fp:
        pickle.dump(obj, fp)
    os.replace(tmp, path)
    chown_default(path)
    return path


def unserialize_from_path(path):
    """Load a pickled object from a file.

    Args:
        path: File path.

    Returns:
        The object.
    """
    with open(path, 'rb') as fp:
        return pickle.load(fp)


_server_globals = {}


def get_ephemeral_content(name: str, init_fn) -> bytes:
    """Get bytes from the ephemeral content cache, creating them if needed.

    The content is stored in the ephemeral directory and created under a server lock,
    so it is shared between processes until the ephemeral cleanup removes it.

    Args:
        name: Content name.
        init_fn: Function without arguments that returns the content.

    Returns:
        The content.
    """
    uid = to_uid(name)
    path = ephemeral_dir('content') + '/' + uid

    def _get():
        if not os.path.isfile(path):
            return
        try:
            return read_file_b(path)
        except OSError:
            return

    b = _get()
    if b is not None:
        return b

    with server_lock(f'ephemeral_content_{uid}'):
        b = _get()
        if b is not None:
            return b

        b = init_fn()
        write_file_b(path, b)
        return b

def get_cached_object(name: str, life_time: int, init_fn):
    """Get an object from the object cache, creating it if needed.

    The object is pickled in ``const.OBJECT_CACHE_DIR`` and created under a server lock.
    A cached object older than ``life_time``, or ``None``, is created again.
    Load and store errors are logged and ignored.

    Args:
        name: Object name.
        life_time: Life time in seconds.
        init_fn: Function without arguments that creates the object.

    Returns:
        The object.
    """
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
    if obj is not None:
        return obj

    with server_lock(uid):
        obj = _get()
        if obj is not None:
            return obj

        obj = init_fn()
        try:
            serialize_to_path(obj, path)
            log.debug(f'get_cached_object {uid!r} - stored')
        except:
            log.exception(f'get_cached_object {uid!r} STORE ERROR')

        return obj


def get_server_global(name: str, init_fn):
    """Get a server-wide global value, creating it if needed.

    The value is kept in memory and pickled in ``const.GLOBALS_DIR``, so other processes load it
    instead of creating it again. It is created under a server lock.
    Load and store errors are logged and ignored.

    Args:
        name: Value name.
        init_fn: Function without arguments that creates the value.

    Returns:
        The value.
    """
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


class LockBusyError(Exception):
    """Raised when a server lock cannot be acquired within the timeout."""

    pass


class _FileLock:
    """Inter-process lock based on ``flock`` on a file in ``const.LOCKS_DIR``."""
    _PAUSE = 0.05

    def __init__(self, uid, timeout):
        """Create the lock.

        Args:
            uid: Lock identifier.
            timeout: Seconds to wait for the lock.
        """
        self.uid = to_uid(uid)
        self.path = const.LOCKS_DIR + '/' + self.uid
        self.timeout = timeout
        self.fp = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def acquire(self):
        """Acquire the lock and write the process id into the lock file.

        Raises:
            LockBusyError: If the lock is not acquired within the timeout.
        """
        ts = time.time()
        self.fp = os.open(self.path, os.O_CREAT | os.O_RDWR)

        while True:
            try:
                fcntl.flock(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                pass
            else:
                os.ftruncate(self.fp, 0)
                os.write(self.fp, str(os.getpid()).encode('ascii'))
                log.debug(f'server lock {self.uid!r}: acquired')
                return

            t = time.time() - ts

            if t >= self.timeout:
                pid = self._holder_pid()
                os.close(self.fp)
                self.fp = None
                log.debug(f'server lock {self.uid!r}: BUSY {pid=}')
                raise LockBusyError(f'server lock {self.uid!r}: busy {pid=}')

            time.sleep(self._PAUSE)

    def release(self):
        """Release the lock. Errors are logged and ignored."""
        if self.fp is None:
            return
        try:
            fcntl.flock(self.fp, fcntl.LOCK_UN)
            os.close(self.fp)
            log.debug(f'server lock {self.uid!r}: released')
        except OSError as exc:
            log.exception(f'server lock {self.uid!r}: RELEASE ERROR {exc!r}')
        self.fp = None

    def _holder_pid(self):
        """Return the process id stored in the lock file, or ``?``."""
        if self.fp is None:
            return '?'
        try:
            os.lseek(self.fp, 0, os.SEEK_SET)
            return os.read(self.fp, 64).decode('ascii') or '?'
        except OSError:
            return '?'


def server_lock(uid, timeout: float = 60):
    """Create an inter-process lock, to be used as a context manager.

    The lock is acquired when the ``with`` block is entered.

    Example::

        with gws.u.server_lock('my_task', timeout=5):
            ...

    Args:
        uid: Lock identifier.
        timeout: Seconds to wait for the lock. ``0`` means a single attempt.

    Returns:
        The lock object.

    Raises:
        LockBusyError: If the lock is not acquired within ``timeout``.
    """
    return _FileLock(uid, timeout)


##


def action_url_path(name: str, **kwargs) -> str:
    """Build a server URL path for an action.

    Example::

        action_url_path('owsService', serviceUid='wms', projectUid='')  # '/_/owsService/serviceUid/wms'

    Args:
        name: Action command name.
        **kwargs: Parameters, appended as ``/key/value`` path segments. Empty values are skipped.

    Returns:
        The URL path.
    """
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
    """Return the Unix time as a float number.

    Returns:
        Seconds since the epoch.
    """
    return time.time()


def stime() -> int:
    """Return the Unix time as an integer number of seconds.

    Returns:
        Seconds since the epoch.
    """
    return int(time.time())


def sleep(n: float):
    """Sleep for a number of seconds.

    Args:
        n: Seconds.
    """
    time.sleep(n)


def mstime() -> int:
    """Return the Unix time as an integer number of milliseconds.

    Returns:
        Milliseconds since the epoch.
    """
    return int(time.time() * 1000)


def microtime() -> int:
    """Return the Unix time as an integer number of microseconds.

    Returns:
        Microseconds since the epoch.
    """
    return int(time.time() * 1000000)
