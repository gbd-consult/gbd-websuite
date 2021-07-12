"""Validate values according to specs"""

import re

import gws.lib.date
import gws.lib.os2
import gws.lib.units
import gws.spec.generator
import gws.types as t
from gws.core.data import Data, is_data_object
from gws.core.types import ExtCommandDescriptor, ExtDescriptor, ISpecRuntime, Params


class Error(Exception):
    pass


def create(options):
    res = gws.spec.generator.generate_for_server(options)
    return Object(res['options'], res['specs'])


class Object(ISpecRuntime):
    def __init__(self, options, specs):
        self.options = options
        self.specs = specs

    def client_vendor_bundle_path(self):
        return self.options.VENDOR_BUNDLE_PATH

    def client_bundle_paths(self):
        paths = []
        for chunk in self.options.chunks:
            path = chunk.bundleDir + '/' + self.options.BUNDLE_FILENAME
            if gws.lib.os2.is_file(path) and path not in paths:
                paths.append(path)
        return paths

    def check_command(self, cmd_name, method, params, strict=True):
        name = method + '.' + cmd_name
        if name not in self.specs:
            return None

        cmd_spec = self.specs[name]
        p = Params(self.read_value(params, cmd_spec.arg, '', strict))

        return ExtCommandDescriptor(
            action_type=cmd_spec.action_type,
            function_name=cmd_spec.function_name,
            class_name=cmd_spec.class_name,
            params=p or Params(),
        )

    def read_value(self, value, type_ref, path='', strict=True, with_error_details=True):
        rd = _Reader(self.specs, path, strict)

        if not with_error_details:
            return _read(rd, value, type_ref)

        rd.stack = [[value, type_ref]]
        try:
            return _read(rd, value, type_ref)
        except Error as e:
            raise _add_error_details(e, path, rd.stack)

    def get_ext_descriptor(self, class_name):
        s = self.specs.get(class_name)
        return t.cast(ExtDescriptor, s) if s else None

    def objects(self, pattern):
        return [s for k, s in self.specs.items() if pattern in k]


##

class _Reader:
    def __init__(self, specs, path, strict):
        self.specs = specs
        self.path = path
        self.strict = strict
        self.stack = None


def _read(rd: _Reader, val, type_ref):
    spec = None
    t = type_ref
    for _ in range(10):
        if isinstance(t, list):
            return _HANDLERS[t[0]](rd, val, t, spec)
        if t in _HANDLERS:
            return _HANDLERS[t](rd, val, t, spec)
        spec = rd.specs.get(t)
        if not spec:
            raise Error('ERR_UNKNOWN', f'unknown type {t!r}', val)
        t = spec.abc
    raise Error('ERR_LOOP', f'unterminated type loop for {type_ref!r}', val)


# type handlers

def _read_any(rd, val, tr, spec):
    return val


def _read_bool(rd, val, tr, spec):
    if rd.strict:
        return _ensure(rd, val, bool)
    try:
        return bool(val)
    except:
        raise Error('ERR_MUST_BE_BOOL', 'must be true or false', val)


def _read_str(rd, val, tr, spec):
    if rd.strict:
        return _ensure(rd, val, str)
    try:
        return _to_string(val)
    except:
        raise Error('ERR_MUST_BE_STRING', 'must be a string', val)


def _read_literal(rd, val, tr, spec):
    s = _read_str(rd, val, tr, spec)
    values = tr[1]
    if s not in values:
        raise Error('ERR_INVALID_CONST', f"expected {_comma(values)}, found {s!r}", val)
    return s


def _read_const(rd, val, tr, spec):
    s = _read_str(rd, val, tr, spec)
    if s != spec.value:
        raise Error('ERR_INVALID_CONST', f"expected {spec.value!r}, found {s!r}", val)
    return s


def _read_bytes(rd, val, tr, spec):
    try:
        if isinstance(val, str):
            return val.encode('utf8', errors='strict')
        return bytes(val)
    except:
        raise Error('ERR_MUST_BE_BYTES', 'must be a byte buffer', val)


def _read_int(rd, val, tr, spec):
    if rd.strict:
        return _ensure(rd, val, int)
    try:
        return int(val)
    except:
        raise Error('ERR_MUST_BE_INT', 'must be an integer', val)


def _read_float(rd, val, tr, spec):
    if rd.strict:
        if isinstance(val, int):
            return float(val)
        return _ensure(rd, val, float)
    try:
        return float(val)
    except:
        raise Error('ERR_MUST_BE_FLOAT', 'must be a float', val)


def _read_list(rd, val, tr, spec):
    if not rd.strict and isinstance(val, str):
        val = val.strip()
        val = [v.strip() for v in val.split(',')] if val else []

    val = _ensure(rd, val, list)
    res = []

    item_type = tr[1]

    for n, v in enumerate(val):
        rd.stack.append([v, n])
        res.append(_read(rd, v, item_type))
        rd.stack.pop()

    return res


def _read_tuple(rd, val, tr, spec):
    if not rd.strict and isinstance(val, str):
        val = val.strip()
        val = [v.strip() for v in val.split(',')] if val else []

    val = _ensure(rd, val, list)

    elts_types = tr[1]

    if len(val) != len(elts_types):
        raise Error('ERR_BAD_TYPE', f"expected {spec.name!r}", val)

    res = []

    for n, v in enumerate(val):
        rd.stack.append([v, n])
        res.append(_read(rd, v, elts_types[n]))
        rd.stack.pop()

    return res


def _read_dict(rd, val, tr, spec):
    return _ensure(rd, val, dict)


def _read_variant(rd, val, tr, spec):
    val = _ensure(rd, val, dict)
    if not rd.strict:
        val = {k.lower(): v for k, v in val.items()}

    # tagged unions are discriminated by 'type'
    # the 'variant' spec is a dict type value => spec key

    type_name = val.get('type')
    if not type_name:
        raise Error('ERR_MISSING_PROP', f"required property missing: 'type'", None)

    variant_types = tr[1]
    target = variant_types.get(type_name)
    if target:
        return _read(rd, val, target)

    raise Error('ERR_BAD_TYPE', f"illegal type: {type_name!r}, expected {_comma(variant_types)}", val)


def _read_union(rd, val, tr, spec):
    # @TODO no untyped unions yet
    raise Error('ERR_BAD_TYPE', 'not supported', val)


def _read_enum(rd, val, tr, spec):
    # NB: our Enums (see __init__) accept both names (for configs) and values (for api calls)
    # this blocks silly things like t.Enum {foo=bar bar=123} but we don't care

    for k, v in spec.values.items():
        if val == k or val == v:
            return v
    raise Error('ERR_BAD_ENUM', f"invalid value, expected {_comma(spec.values)}", val)


def _read_object(rd, val, tr, spec):
    val = _ensure(rd, val, dict)
    if not rd.strict:
        val = {k.lower(): v for k, v in val.items()}

    res = {}

    for prop_name, prop_key in spec.props.items():
        prop_val = val.get(prop_name if rd.strict else prop_name.lower())
        rd.stack.append([prop_val, prop_name])
        res[prop_name] = _read(rd, prop_val, prop_key)
        rd.stack.pop()

    if rd.strict:
        unknown = [k for k in val if k not in spec.props]
        if unknown:
            w = 'property' if len(unknown) == 1 else 'properties'
            raise Error('ERR_UNKNOWN_PROP', f"unknown {w}: {_comma(unknown)}, expected {_comma(spec.props)}", val)

    return Data(res)


def _read_property(rd, val, tr, spec):
    if val is not None:
        return _read(rd, val, spec.type)
    if not spec.has_default:
        raise Error('ERR_MISSING_PROP', f"required property missing: {spec.ident!r}", None)
    if spec.default is None:
        return None
    # the default, if given, must match the type
    # NB, for Data objects, default={} will create an objects with defaults
    return _read(rd, spec.default, spec.type)


def _read_alias(rd, val, tr, spec):
    return _read(rd, val, spec.target)


def _read_dirpath(rd, val, tr, spec):
    path = gws.lib.os2.abs_path(val, rd.path)
    if not gws.lib.os2.is_dir(path):
        raise Error('ERR_DIR_NOT_FOUND', 'directory not found', path)
    return path


def _read_filepath(rd, val, tr, spec):
    path = gws.lib.os2.abs_path(val, rd.path)
    if not gws.lib.os2.is_file(path):
        raise Error('ERR_FILE_NOT_FOUND', 'file not found', path)
    return path


def _read_duration(rd, val, tr, spec):
    try:
        return gws.lib.units.parse_duration(val)
    except ValueError:
        raise Error('ERR_BAD_DURATION', 'invalid duration', val)


def _read_regex(rd, val, tr, spec):
    try:
        re.compile(val)
        return val
    except re.error as e:
        raise Error('ERR_BAD_REGEX', f"invalid regular expression: {e!r}", val)


def _read_formatstr(rd, val, tr, spec):
    # @TODO
    return _read_str(rd, val, tr, spec)


def _read_crs(rd, val, tr, spec):
    # @TODO: crs validation
    return _read_str(rd, val, tr, spec)


def _read_color(rd, val, tr, spec):
    # @TODO: color validation
    return _read_str(rd, val, tr, spec)


def _read_date(rd, val, tr, spec):
    d = gws.lib.date.from_iso(str(val))
    if not d:
        raise Error('ERR_INVALID_DATE', 'invalid date', val)
    return gws.lib.date.to_iso_date(d)


def _read_datetime(rd, val, tr, spec):
    d = gws.lib.date.from_iso(str(val))
    if not d:
        raise Error('ERR_INVALID_DATE', 'invalid date', val)
    return gws.lib.date.to_iso(d)


def _read_url(rd, val, tr, spec):
    # @TODO: url validation
    return _read_str(rd, val, tr, spec)


# utils


def _ensure(rd, val, klass):
    if isinstance(val, klass):
        return val
    if klass == list and isinstance(val, tuple):
        return list(val)
    if klass == dict and is_data_object(val):
        return vars(val)
    raise Error('ERR_WRONG_TYPE', f"wrong type {_classname(type(val))!r}, expected {_classname(klass)!r}", val)


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
    'str': _read_str,
    'literal': _read_literal,
    'const': _read_const,
    'tuple': _read_tuple,
    'variant': _read_variant,
    'union': _read_union,

    'object': _read_object,
    'property': _read_property,
    'alias': _read_alias,

    'gws.Crs': _read_crs,
    'gws.Color': _read_color,
    'gws.Date': _read_date,
    'gws.DateTime': _read_datetime,
    'gws.DirPath': _read_dirpath,
    'gws.Duration': _read_duration,
    'gws.FilePath': _read_filepath,
    'gws.FormatStr': _read_formatstr,
    'gws.Regex': _read_regex,
    'gws.Url': _read_url,

    't.Any': _read_any,

}


def _add_error_details(e: Error, path: str, stack):
    details = {}

    try:
        details['formatted_value'] = _format_value(e.args[2])
    except:
        details['formatted_value'] = ''

    details['path'] = path

    details['stack'] = stack or []
    details['formatted_stack'] = _format_stack(details['stack'])

    e.args = e.args + tuple([details])
    return e


def _format_value(val):
    s = repr(val)
    if len(s) > 600:
        s = s[:600] + '...'
    return s


def _format_stack(stack):
    f = []

    for val, name in stack:
        name = repr(name)
        line = 'item ' + name if name.isdigit() else name
        for p in 'uid', 'title', 'type':
            try:
                s = val.get(p)
                if s is not None:
                    line += f' ({p}={s!r})'
                    break
            except:
                pass
        f.append('in ' + line)

    return '\n'.join(f)
