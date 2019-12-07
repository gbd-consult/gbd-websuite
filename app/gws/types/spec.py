"""Validate values according to specs"""

import os
import re
import gws.tools.misc as misc
from .data import Data


class Error(Exception):
    pass


class Validator:
    def __init__(self, spec):
        self.spec = spec

    def method_spec(self, name):
        return self.spec.get('method:' + name)

    def read_value(self, val, type_name, path='', strict=True):
        reader = _Reader(self.spec, path, strict)
        return reader.read(val, type_name)


##

class _Reader:
    def __init__(self, spec, path, strict):
        self.spec = spec
        self.path = path
        self.keys = []
        self.strict = strict
        self.handlers = _HANDLERS

    def error(self, code, msg, val):
        val = repr(val)
        if len(val) > 600:
            val = val[:600] + '...'
        raise Error(code + ': ' + msg, self.path, '.'.join(str(k) for k in self.keys), val)

    def read(self, val, type_name):
        if type_name in self.handlers:
            return self.handlers[type_name](self, val, None)
        s = self.spec[type_name]
        return self.handlers[s['type']](self, val, s)


## type handlers    

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


def _read_bytes(rd, val, spec):
    try:
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
        rd.keys.append(name)
        try:
            res[name] = _property_value(rd, val.get(name if rd.strict else name.lower()), p)
        finally:
            rd.keys.pop()

    if rd.strict:
        names = set(p['name'] for p in spec['props'])
        unknown = [key for key in val if key not in names]
        if len(unknown) == 1:
            return rd.error('ERR_UNKNOWN_PROP', 'unknown property: %r' % unknown[0], val)
        if len(unknown) > 1:
            return rd.error('ERR_UNKNOWN_PROP', 'unknown properties: %r' % ', '.join(unknown), val)

    return Data(res)


def _read_typeunion(rd, val, spec):
    if not hasattr(val, 'get'):
        return rd.error('ERR_UNEXPECTED', 'unexpected value', val)

    # type unions are discriminated by the "type" prop
    # e.g {type:printer} will select the base "gws.ext.action.printer.Config"

    type_name = val.get('type')

    if type_name:
        base = None
        for b in spec['bases']:
            p = b.split('.')
            if p[-2] == type_name:
                base = b
                break

        if base:
            return rd.read(val, base)

    return rd.error('ERR_BAD_TYPE', 'expected %r' % ' or '.join(spec['bases']), val)


def _read_union(rd, val, spec):
    # @TODO no untyped unions yet
    return rd.error('ERR_BAD_TYPE', 'not supported', val)


def _read_list(rd, val, spec):
    if not rd.strict and isinstance(val, str):
        val = val.split(',')

    val = _ensure(rd, val, list)
    res = []

    for n, v in enumerate(val):
        rd.keys.append(n)
        try:
            res.append(rd.read(v, spec['base']))
        finally:
            rd.keys.pop()

    return res


def _read_tuple(rd, val, spec):
    if not rd.strict and isinstance(val, str):
        val = val.split(',')

    val = _ensure(rd, val, list)

    if len(val) != len(spec['bases']):
        rd.error('ERR_BAD_TYPE', 'expected %r' % spec['name'], val)

    res = []

    for n, v in enumerate(val):
        rd.keys.append(n)
        try:
            res.append(rd.read(v, spec['bases'][n]))
        finally:
            rd.keys.pop()

    return res


def _read_enum(rd, val, spec):
    # NB: our Enums (see __init__) accept both names (for configs) and values (for api calls)
    # this blocks silly things like Enum {foo=bar bar=123} but we don't care

    for k, v in spec['values'].items():
        if val == k or val == v:
            return v
    rd.error('ERR_BAD_ENUM', 'invalid value (expected %r)' % ' or '.join(v for v in spec['values']), val)


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
        return misc.parse_duration(val)
    except ValueError:
        rd.error('ERR_BAD_DURATION', 'invalid duration', val)


def _read_regex(rd, val, spec):
    try:
        re.compile(val)
        return val
    except re.error as e:
        rd.error('ERR_BAD_REGEX', 'invalid regular expression: %r' % e, val)


def _read_formatstr(rd, val, spec):
    # @TODO
    return _read_str(rd, val, spec)


def _read_crsref(rd, val, spec):
    # @TODO: crs validation
    return _read_str(rd, val, spec)


def _read_date(rd, val, spec):
    # @TODO: date validation
    return _read_str(rd, val, spec)


def _read_url(rd, val, spec):
    # @TODO: url validation
    return _read_str(rd, val, spec)


## utils

def _property_value(rd, prop_val, spec):
    default = spec['default']

    # no value?

    if prop_val is None:
        if not spec['optional']:
            return rd.error('ERR_MISSING_PROP', 'required property missing: %r' % spec['name'], 'nothing')

        # no default as well
        if default is None:
            return None

        # the default, if given, must match the type
        # NB, for Data objects, default={} will create an objects with defaults
        return rd.read(default, spec['type'])

    # have value...

    # @TODO we don't have literals for now
    #
    # if spec['type'] == 'gws.types.literal':
    #     # if literal, the value must be exactly spec[default]
    #     if prop_val != default:
    #         return rd.error('ERR_BAD_LITERAL', 'expected %r' % default, prop_val)
    #     return prop_val

    # no literal, normal case - the value is given and must match the type

    return rd.read(prop_val, spec['type'])


def _ensure(rd, val, klass):
    if isinstance(val, klass):
        return val
    if klass == list and isinstance(val, tuple):
        return list(val)
    if klass == dict and isinstance(val, Data):
        return val.as_dict()
    if isinstance(klass, type):
        klass = 'object' if klass == dict else klass.__name__
    rd.error('ERR_WRONG_TYPE', '%r expected' % klass, val)


def _to_string(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return x.decode('utf8')
    raise ValueError()


##

_HANDLERS = {
    'bool': _read_bool,
    'bytes': _read_bytes,
    'dict': _read_dict,
    'enum': _read_enum,
    'float': _read_float,
    'int': _read_int,
    'list': _read_list,
    'object': _read_object,
    'str': _read_str,
    'tuple': _read_tuple,
    'typeunion': _read_typeunion,
    'union': _read_union,

    'gws.types.crsref': _read_crsref,
    'gws.types.date': _read_date,
    'gws.types.dirpath': _read_dirpath,
    'gws.types.duration': _read_duration,
    'gws.types.filepath': _read_filepath,
    'gws.types.formatstr': _read_formatstr,
    'gws.types.regex': _read_regex,
    'gws.types.url': _read_url,

}
