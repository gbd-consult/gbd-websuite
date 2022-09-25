"""Read and validate values according to spec types."""

import re

import gws
import gws.gis.crs
import gws.lib.date
import gws.lib.os2
import gws.lib.units

from . import core


class Reader:
    stack = None
    push = lambda _: ...
    pop = lambda: ...
    atom = core.Type(c=core.C.ATOM)

    def __init__(self, runtime, path, strict_mode, verbose_errors, accept_extra_props):
        self.runtime = runtime
        self.path = path
        self.strict_mode = strict_mode
        self.verbose_errors = verbose_errors
        self.accept_extra_props = accept_extra_props

    def read(self, value, type_uid):
        if not self.verbose_errors:
            return self.read2(value, type_uid)

        self.stack = [[value, type_uid]]
        self.push = self.stack.append
        self.pop = self.stack.pop

        try:
            return self.read2(value, type_uid)
        except core.ReadError as exc:
            raise self.add_error_details(exc)

    def read2(self, value, type_uid):
        typ = self.runtime.get(type_uid)

        if type_uid in _READERS:
            return _READERS[type_uid](self, value, typ or self.atom)

        if not typ:
            raise core.ReadError(f'unknown type {type_uid!r}', value)

        if typ.c not in _READERS:
            raise core.ReadError(f'unknown type category {typ.c!r}', value)

        return _READERS[typ.c](self, value, typ)

    def add_error_details(self, exc: Exception):
        details = {
            'formatted_value': _format_error_value(exc),
            'path': self.path,
            'formatted_stack': _format_error_stack(self.stack or [])
        }
        exc.args = exc.args + tuple([details])
        return exc


# atoms

def _read_any(r: Reader, val, typ: core.Type):
    return val


def _read_bool(r: Reader, val, typ: core.Type):
    if r.strict_mode:
        return _ensure(val, bool)
    try:
        return bool(val)
    except:
        raise core.ReadError('must be true or false', val)


def _read_bytes(r: Reader, val, typ: core.Type):
    try:
        if isinstance(val, str):
            return val.encode('utf8', errors='strict')
        return bytes(val)
    except:
        raise core.ReadError('must be a byte buffer', val)


def _read_float(r: Reader, val, typ: core.Type):
    if r.strict_mode:
        if isinstance(val, int):
            return float(val)
        return _ensure(val, float)
    try:
        return float(val)
    except:
        raise core.ReadError('must be a float', val)


def _read_int(r: Reader, val, typ: core.Type):
    if r.strict_mode:
        return _ensure(val, int)
    try:
        return int(val)
    except:
        raise core.ReadError('must be an integer', val)


def _read_str(r: Reader, val, typ: core.Type):
    if r.strict_mode:
        return _ensure(val, str)
    try:
        return _to_string(val)
    except:
        raise core.ReadError('must be a string', val)


# built-ins

def _read_raw_dict(r: Reader, val, typ: core.Type):
    return _ensure(val, dict)


def _read_dict(r: Reader, val, typ: core.Type):
    dct = {}
    for k, v in _ensure(val, dict).items():
        dct[k] = r.read2(v, typ.tValue)
    return dct


def _read_raw_list(r: Reader, val, typ: core.Type):
    return _ensure(val, list)


def _read_list(r: Reader, val, typ: core.Type):
    lst = _read_any_list(r, val)
    res = []
    for n, v in enumerate(lst):
        r.push([v, n])
        res.append(r.read2(v, typ.tItem))
        r.pop()
    return res


def _read_set(r: Reader, val, typ: core.Type):
    lst = _read_list(r, val, typ)
    return set(lst)


def _read_tuple(r: Reader, val, typ: core.Type):
    lst = _read_any_list(r, val)

    if len(lst) != len(typ.tItems):
        raise core.ReadError(f"expected: {_comma(typ.tItems)}", val)

    res = []
    for n, v in enumerate(lst):
        r.push([v, n])
        res.append(r.read2(v, typ.tItems[n]))
        r.pop()
    return res


def _read_any_list(r, val):
    if not r.strict_mode and isinstance(val, str):
        val = val.strip()
        val = [v.strip() for v in val.split(',')] if val else []
    return _ensure(val, list)


def _read_literal(r: Reader, val, typ: core.Type):
    s = _read_any(r, val, typ)
    if s not in typ.literalValues:
        raise core.ReadError(f"invalid value: {s!r}, expected: {_comma(typ.literalValues)}", val)
    return s


def _read_optional(r: Reader, val, typ: core.Type):
    if val is None:
        return val
    return r.read2(val, typ.tTarget)


def _read_union(r: Reader, val, typ: core.Type):
    # @TODO no untyped unions yet
    raise core.ReadError('unions are not supported yet', val)


# our types

def _read_type(r: Reader, val, typ: core.Type):
    return r.read2(val, typ.tTarget)


def _read_enum(r: Reader, val, typ: core.Type):
    # NB: our Enums accept both names (for configs) and values (for api calls)
    # this prevents silly things like Enum{foo=bar bar=123} but we don't care
    #
    # this reader returns a value, it's up to the caller to convert it to the actual enum

    for k, v in typ.enumValues.items():
        if val == k or val == v:
            return v
    raise core.ReadError(f"invalid value: {val!r}, expected: {_comma(typ.enumValues)}", val)


def _read_object(r: Reader, val, typ: core.Type):
    val = _ensure(val, dict)
    if not r.strict_mode:
        val = {k.lower(): v for k, v in val.items()}

    res = {}

    for prop_name, prop_type_uid in typ.tProperties.items():
        prop_val = val.get(prop_name if r.strict_mode else prop_name.lower())
        r.push([prop_val, prop_name])
        res[prop_name] = r.read2(prop_val, prop_type_uid)
        r.pop()

    unknown = []

    for k in val:
        if k not in typ.tProperties:
            if r.accept_extra_props:
                res[k] = val[k]
            elif r.strict_mode:
                unknown.append(k)

    if unknown:
        raise core.ReadError(f"unknown keys: {_comma(unknown)}, expected: {_comma(typ.tProperties)}", val)

    return gws.Data(res)


def _read_property(r: Reader, val, typ: core.Type):
    if val is not None:
        return r.read2(val, typ.tValue)

    if not typ.hasDefault:
        raise core.ReadError(f"required property missing: {typ.ident!r}", None)

    if typ.default is None:
        return None

    # the default, if given, must match the type
    # NB, for Data objects, default={} will create an object with defaults
    return r.read2(typ.default, typ.tValue)


def _read_variant(r: Reader, val, typ: core.Type):
    val = _ensure(val, dict)
    if not r.strict_mode:
        val = {k.lower(): v for k, v in val.items()}

    type_name = val.get('type', core.DEFAULT_TYPE)
    target_type_uid = typ.tMembers.get(type_name)
    if not target_type_uid:
        raise core.ReadError(f"illegal type: {type_name!r}, expected: {_comma(typ.tMembers)}", val)
    return r.read2(val, target_type_uid)


# custom types

def _read_acl(r: Reader, val, typ: core.Type):
    v = _read_acl2(val)
    if v is None:
        raise core.ReadError(f'invalid ACL: {val!r}', val)
    return v


def _read_acl2(val):
    # for ACLs we accept a string like "allow foo, deny bar"
    # or a list of dicts [ {type:allow, role:foo}, {type:deny, role:bar}]

    ps = []

    if isinstance(val, str):
        for elem in val.split(','):
            elem = elem.split()
            if len(elem) != 2:
                return
            ps.append([elem[0].lower(), elem[1]])
    elif isinstance(val, list):
        for elem in val:
            if not isinstance(elem, dict) or len(elem) != 2 or 'type' not in elem or 'role' not in elem:
                return
            ps.append([elem['type'].lower(), elem['role']])
    else:
        return

    res = []
    for a, r in ps:
        if a == 'allow' and r.isalnum():
            res.append([gws.ACCESS_ALLOWED, r])
        elif a == 'deny' and r.isalnum():
            res.append([gws.ACCESS_DENIED, r])
        else:
            return
    return res


def _read_color(r: Reader, val, typ: core.Type):
    # @TODO: parse color values
    return _read_str(r, val, typ)


def _read_crs(r: Reader, val, typ: core.Type):
    crs = gws.gis.crs.get(val)
    if not crs:
        raise core.ReadError(f'invalid crs: {val!r}', val)
    return crs.srid


def _read_date(r: Reader, val, typ: core.Type):
    try:
        d = gws.lib.date.from_iso(str(val))
    except ValueError:
        raise core.ReadError(f'invalid date: {val!r}', val)
    return gws.lib.date.to_iso_date(d)


def _read_datetime(r: Reader, val, typ: core.Type):
    try:
        d = gws.lib.date.from_iso(str(val))
    except ValueError:
        raise core.ReadError(f'invalid date: {val!r}', val)
    return gws.lib.date.to_iso(d)


def _read_dirpath(r: Reader, val, typ: core.Type):
    path = gws.lib.os2.abs_path(val, r.path)
    if not gws.is_dir(path):
        raise core.ReadError(f'directory not found: {path!r}', path)
    return path


def _read_duration(r: Reader, val, typ: core.Type):
    try:
        return gws.lib.units.parse_duration(val)
    except ValueError:
        raise core.ReadError(f'invalid duration: {val!r}', val)


def _read_filepath(r: Reader, val, typ: core.Type):
    path = gws.lib.os2.abs_path(val, r.path)
    if not gws.is_file(path):
        raise core.ReadError(f'file not found: {path!r}', path)
    return path


def _read_formatstr(r: Reader, val, typ: core.Type):
    # @TODO validate
    return _read_str(r, val, typ)


def _read_regex(r: Reader, val, typ: core.Type):
    try:
        re.compile(val)
        return val
    except re.error as e:
        raise core.ReadError(f'invalid regular expression: {val!r}: {e!r}', val)


def _read_url(r: Reader, val, typ: core.Type):
    u = _read_str(r, val, typ)
    if u.startswith(('http://', 'https://')):
        return u
    raise core.ReadError(f'invalid url: {val!r}', val)


# utils


def _ensure(val, cls):
    if isinstance(val, cls):
        return val
    if cls == list and isinstance(val, tuple):
        return list(val)
    if cls == dict and gws.is_data_object(val):
        return vars(val)
    raise core.ReadError(f"wrong type: {_classname(type(val))!r}, expected: {_classname(cls)!r}", val)


def _to_string(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return x.decode('utf8')
    raise ValueError()


def _classname(cls):
    try:
        return cls.__name__
    except:
        return str(cls)


def _comma(ls):
    return repr(', '.join(sorted(str(x) for x in ls)))


##


def _format_error_value(exc):
    try:
        val = exc.args[1]
    except (AttributeError, IndexError):
        return ''

    s = repr(val)
    if len(s) > 600:
        s = s[:600] + '...'
    return s


def _format_error_stack(stack):
    f = []

    for val, name in reversed(stack):
        name = repr(name)
        line = 'item ' + name if name.isdigit() else name
        for p in 'uid', 'title', 'type':
            try:
                s = val.get(p)
                if s is not None:
                    line += f' ({p}={s!r})'
                    break
            except AttributeError:
                pass
        f.append('in ' + line)

    return f


# 

_READERS = {
    'any': _read_any,
    'bool': _read_bool,
    'bytes': _read_bytes,
    'float': _read_float,
    'int': _read_int,
    'str': _read_str,

    'list': _read_raw_list,
    'dict': _read_raw_dict,

    core.C.CLASS: _read_object,
    core.C.DICT: _read_dict,
    core.C.ENUM: _read_enum,
    core.C.LIST: _read_list,
    core.C.LITERAL: _read_literal,
    core.C.OPTIONAL: _read_optional,
    core.C.PROPERTY: _read_property,
    core.C.SET: _read_set,
    core.C.TUPLE: _read_tuple,
    core.C.TYPE: _read_type,
    core.C.UNION: _read_union,
    core.C.VARIANT: _read_variant,
    core.C.CONFIG: _read_object,
    core.C.PROPS: _read_object,

    'gws.core.types.ACL': _read_acl,
    'gws.core.types.Color': _read_color,
    'gws.core.types.CrsName': _read_crs,
    'gws.core.types.Date': _read_date,
    'gws.core.types.DateTime': _read_datetime,
    'gws.core.types.DirPath': _read_dirpath,
    'gws.core.types.Duration': _read_duration,
    'gws.core.types.FilePath': _read_filepath,
    'gws.core.types.FormatStr': _read_formatstr,
    'gws.core.types.Regex': _read_regex,
    'gws.core.types.Url': _read_url,
}
