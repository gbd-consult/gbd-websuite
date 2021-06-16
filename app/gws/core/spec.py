"""Validate values according to specs"""

import os
import re
import gws.lib.units
import gws.lib.date

import gws.types as t


class Error(Exception):
    def __init__(self, *args):
        super().__init__(*args)
        self.message = args[0] if args else ''


#:export SpecValidator
class SpecValidator:
    def __init__(self, spec):
        self.spec = spec

    def method_spec(self, name):
        return self.spec.get('method:' + name)

    def read_value(self, val, type_name, path='', strict=True):
        reader = _Reader(self.spec, path, strict)
        return reader.read_main(val, type_name)


##

class _Reader:
    def __init__(self, spec, path, strict):
        self.spec = spec
        self.path = path
        self.stack = []
        self.strict = strict
        self.handlers = _HANDLERS

    def error(self, code, msg, val):
        def _short(v):
            v = repr(v)
            if len(v) > 600:
                v = v[:600] + '...'
            return v

        def _item(name, v):
            m = repr(name)
            if m.isdigit():
                m = 'item ' + m
            m = 'in ' + m

            for p in 'uid', 'title', 'type':
                try:
                    s = v.get(p)
                    if s is not None:
                        return f'{m} ({p}={s!r})'
                except:
                    pass

            return m

        raise Error(
            code + ': ' + msg,
            self.path,
            '\n'.join(_item(name, v) for name, v in self.stack),
            _short(val))

    def read_main(self, val, type_name):
        self.stack.append((type_name, val))
        try:
            return self.read(val, type_name)
        finally:
            self.stack.pop()

    def read(self, val, type_name):
        if type_name in self.handlers:
            return self.handlers[type_name](self, val, None)
        s = self.spec[type_name]
        return self.handlers[s['type']](self, val, s)


# type handlers

def _read_any(rd, val, spec):
    return val


def _read_bool(rd, val, spec):
    if rd.strict:
        return _ensure(rd, val, bool)
    try:
        return bool(val)
    except:
        rd.error('ERR_MUST_BE_BOOL', 'must be true or false', val)


def _read_str(rd, val, spec):
    if rd.strict:
        return _ensure(rd, val, str)
    try:
        return _to_string(val)
    except:
        rd.error('ERR_MUST_BE_STRING', 'must be a string', val)


def _read_literal(rd, val, spec):
    return _read_str(rd, val, spec)


def _read_bytes(rd, val, spec):
    try:
        if isinstance(val, str):
            return val.encode('utf8', errors='strict')
        return bytes(val)
    except:
        rd.error('ERR_MUST_BE_BYTES', 'must be a byte buffer', val)


def _read_int(rd, val, spec):
    if rd.strict:
        return _ensure(rd, val, int)
    try:
        return int(val)
    except:
        rd.error('ERR_MUST_BE_INT', 'must be an integer', val)


def _read_float(rd, val, spec):
    if rd.strict:
        if isinstance(val, int):
            return float(val)
        return _ensure(rd, val, float)
    try:
        return float(val)
    except:
        rd.error('ERR_MUST_BE_FLOAT', 'must be a float', val)


def _read_dict(rd, val, spec):
    return _ensure(rd, val, dict)


def _read_object(rd, val, spec):
    val = _ensure(rd, val, dict)
    if not rd.strict:
        val = {k.lower(): v for k, v in val.items()}

    res = {}

    for p in spec['props']:
        name = p['name']
        pval = val.get(name if rd.strict else name.lower())
        rd.stack.append((name, pval))
        try:
            res[name] = _property_value(rd, pval, p)
        finally:
            rd.stack.pop()

    if rd.strict:
        names = set(p['name'] for p in spec['props'])
        unknown = [k for k in val if k not in names]
        if unknown:
            w = 'property' if len(unknown) == 1 else 'properties'
            return rd.error('ERR_UNKNOWN_PROP', f"unknown {w}: {_comma(unknown)}, expected {_comma(names)}", val)

    return t.Data(res)


def _read_taggedunion(rd, val, spec):
    if not hasattr(val, 'get'):
        return rd.error('ERR_UNEXPECTED', 'unexpected value', val)

    # tagged unions are discriminated by the "tag" prop
    # the 'parts' spec is a dict tag_value => class_name

    type_name = val.get(spec['tag'])
    base = spec['parts'].get(type_name)

    if base:
        return rd.read(val, base)

    return rd.error('ERR_BAD_TYPE', f"illegal type: {type_name!r}, expected {_comma(spec['parts'])}", val)


def _read_union(rd, val, spec):
    # @TODO no untyped unions yet
    return rd.error('ERR_BAD_TYPE', 'not supported', val)


def _read_list(rd, val, spec):
    if not rd.strict and isinstance(val, str):
        val = val.strip()
        val = [v.strip() for v in val.split(',')] if val else []

    val = _ensure(rd, val, list)
    res = []

    for n, v in enumerate(val):
        rd.stack.append((n, v))
        try:
            res.append(rd.read(v, spec['bases'][0]))
        finally:
            rd.stack.pop()

    return res


def _read_tuple(rd, val, spec):
    if not rd.strict and isinstance(val, str):
        val = val.strip()
        val = [v.strip() for v in val.split(',')] if val else []

    val = _ensure(rd, val, list)

    if len(val) != len(spec['bases']):
        rd.error('ERR_BAD_TYPE', f"expected {spec['name']!r}", val)

    res = []

    for n, v in enumerate(val):
        rd.stack.append((n, v))
        try:
            res.append(rd.read(v, spec['bases'][n]))
        finally:
            rd.stack.pop()

    return res


def _read_enum(rd, val, spec):
    # NB: our Enums (see __init__) accept both names (for configs) and values (for api calls)
    # this blocks silly things like Enum {foo=bar bar=123} but we don't care

    for k, v in spec['values'].items():
        if val == k or val == v:
            return v
    rd.error('ERR_BAD_ENUM', f"invalid value, expected {_comma(spec['values'])}", val)


def _read_dirpath(rd, val, spec):
    path = os.path.join(os.path.dirname(rd.path), val)
    if not os.path.isdir(path):
        rd.error('ERR_DIR_NOT_FOUND', 'directory not found', path)
    return path


def _read_filepath(rd, val, spec):
    path = os.path.join(os.path.dirname(rd.path), val)
    if not os.path.isfile(path):
        rd.error('ERR_FILE_NOT_FOUND', 'file not found', path)
    return path


def _read_duration(rd, val, spec):
    try:
        return gws.lib.units.parse_duration(val)
    except ValueError:
        rd.error('ERR_BAD_DURATION', 'invalid duration', val)


def _read_regex(rd, val, spec):
    try:
        re.compile(val)
        return val
    except re.error as e:
        rd.error('ERR_BAD_REGEX', f"invalid regular expression: {e!r}", val)


def _read_formatstr(rd, val, spec):
    # @TODO
    return _read_str(rd, val, spec)


def _read_crs(rd, val, spec):
    # @TODO: crs validation
    return _read_str(rd, val, spec)


def _read_color(rd, val, spec):
    # @TODO: color validation
    return _read_str(rd, val, spec)


def _read_date(rd, val, spec):
    d = gws.lib.date.from_iso(str(val))
    if not d:
        return rd.error('ERR_INVALID_DATE', 'invalid date', val)
    return gws.lib.date.to_iso_date(d)


def _read_datetime(rd, val, spec):
    d = gws.lib.date.from_iso(str(val))
    if not d:
        return rd.error('ERR_INVALID_DATE', 'invalid date', val)
    return gws.lib.date.to_iso(d)


def _read_url(rd, val, spec):
    # @TODO: url validation
    return _read_str(rd, val, spec)


## utils

def _property_value(rd, prop_val, spec):
    default = spec['default']

    # no value?

    if prop_val is None:
        if not spec['optional'] and default is not None:
            return rd.error('ERR_MISSING_PROP', f"required property missing: {spec['name']!r}", 'nothing')

        # no default as well
        if default is None:
            return None

        # the default, if given, must match the type
        # NB, for Data objects, default={} will create an objects with defaults
        return rd.read(default, spec['type'])

    return rd.read(prop_val, spec['type'])


def _ensure(rd, val, klass):
    if isinstance(val, klass):
        return val
    if klass == list and isinstance(val, tuple):
        return list(val)
    if klass == dict and gws.is_data_object(val):
        return vars(val)
    rd.error('ERR_WRONG_TYPE', f"wrong type {_classname(type(val))!r}, expected {_classname(klass)!r}", val)


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
    return ', '.join(sorted(repr(x) for x in ls))


##

_HANDLERS = {
    'any': _read_any,
    'bool': _read_bool,
    'bytes': _read_bytes,
    'dict': _read_dict,
    'enum': _read_enum,
    'float': _read_float,
    'int': _read_int,
    'list': _read_list,
    'object': _read_object,
    'str': _read_str,
    'literal': _read_literal,
    'tuple': _read_tuple,
    'taggedunion': _read_taggedunion,

    'gws.types.Crs': _read_crs,
    'gws.types.Color': _read_color,
    'gws.types.Date': _read_date,
    'gws.types.DateTime': _read_datetime,
    'gws.types.DirPath': _read_dirpath,
    'gws.types.Duration': _read_duration,
    'gws.types.FilePath': _read_filepath,
    'gws.types.FormatStr': _read_formatstr,
    'gws.types.Regex': _read_regex,
    'gws.types.Url': _read_url,

    'gws.types.Any': _read_any,

}
