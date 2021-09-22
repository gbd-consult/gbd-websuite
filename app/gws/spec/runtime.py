"""Validate values according to specs"""

import re

import gws.lib.date
import gws.lib.json2
import gws.lib.os2
import gws.lib.units
import gws.spec.generator
from gws.core.data import Data, is_data_object
from gws.core.types import ExtCommandDescriptor, ExtObjectDescriptor, ISpecRuntime, Params


class Error(Exception):
    pass


def create(manifest_path: str = None) -> 'Object':
    genres = _generate(manifest_path)
    return Object(genres)


def create_and_store(manifest_path: str = None) -> 'Object':
    genres = _generate(manifest_path)
    cc = _cache_path(manifest_path)

    try:
        gws.lib.json2.to_path(cc, genres, pretty=True)
        gws.log.debug(f'spec.create: stored to {cc!r}')
    except:
        gws.log.exception(f'spec.create: store failed')

    return Object(genres)


def load(manifest_path: str = None) -> 'Object':
    cc = _cache_path(manifest_path)

    if gws.is_file(cc):
        try:
            genres = gws.lib.json2.from_path(cc)
            gws.log.debug(f'spec.load: loaded from {cc!r}')
            return Object(genres)
        except:
            gws.log.exception(f'spec.load: load failed')

    return create_and_store(manifest_path)


def _generate(manifest_path):
    ts = gws.time_start('SPEC GENERATOR')
    try:
        genres = gws.spec.generator.generate_for_server(manifest_path)
    except Exception as exc:
        raise Error(f'system error, spec generator failed') from exc
    gws.time_end(ts)
    return genres


def _cache_path(manifest_path):
    return gws.TMP_DIR + '/spec_' + gws.sha256(manifest_path or '') + '.json'


##


class Object(ISpecRuntime):
    def __init__(self, genres):
        self.meta = genres['meta']
        self.manifest = gws.Manifest(self.meta['manifest'])
        self.specs = genres['specs']
        self.strings = genres['strings']

    def client_vendor_bundle_path(self):
        return self.meta['VENDOR_BUNDLE_PATH']

    def client_bundle_paths(self):
        paths = []
        for chunk in self.meta['chunks']:
            path = chunk['bundleDir'] + '/' + self.meta['BUNDLE_FILENAME']
            if gws.is_file(path) and path not in paths:
                paths.append(path)
        return paths

    def check_command(self, cmd_name, cmd_method, params, strict=True):
        name = cmd_method + '.' + cmd_name
        if name not in self.specs:
            return None

        cmd_spec = self.specs[name]
        p = Params(self.read_value(params, cmd_spec['arg'], '', strict, with_error_details=False))

        return ExtCommandDescriptor(
            class_name=cmd_spec['class_name'],
            cmd_action=cmd_spec['cmd_action'],
            cmd_name=cmd_spec['cmd_name'],
            function_name=cmd_spec['function_name'],
            params=p or Params(),
        )

    def ext_object_descriptor(self, class_name):
        s = self.specs.get(class_name)
        return ExtObjectDescriptor(s) if s else None

    def cli_docs(self, lang):
        strings = self.strings.get(lang) or self.strings['en']
        docs = []

        for spec in self.specs.values():
            if spec.get('cmd_method') != 'cli':
                continue

            tab = ' ' * 4
            doc = f"gws {spec['cmd_action']} {spec['cmd_command']}\n{tab}{strings.get(spec['name'], '')}\n"
            args = []

            for name, key in self.specs[spec['arg']]['props'].items():
                prop_spec = self.specs[key]
                ln = f"{tab}{tab}--{name:15} {strings.get(prop_spec['name'], '')}"
                if not gws.is_empty(prop_spec['default']):
                    ln += f" (default {prop_spec['default']})"
                if not prop_spec['has_default']:
                    ln += " (*)"
                args.append(ln)

            if args:
                doc += '\n'.join(sorted(args))
                doc += '\n'

            docs.append([spec['cmd_action'], spec['cmd_command'], doc])

        return docs

    def read_value(self, value, type_name, path='', strict=True, with_error_details=True):
        stack = [[value, type_name]] if with_error_details else None
        rd = _Reader(self.specs, path, strict, stack)

        if not with_error_details:
            return rd.read(value, type_name)

        try:
            return rd.read(value, type_name)
        except Error as e:
            raise _add_error_details(e, path, rd.stack)


##

class _Reader:
    _handlers = {
        'any': '_read_any',
        'bool': '_read_bool',
        'bytes': '_read_bytes',
        'float': '_read_float',
        'int': '_read_int',
        'str': '_read_str',

        'TDict': '_read_dict',
        'TList': '_read_list',
        'TLiteral': '_read_literal',
        'TOptional': '_read_optional',
        'TTuple': '_read_tuple',
        'TUnion': '_read_union',
        'TVariant': '_read_variant',

        'TAlias': '_read_alias',
        'TEnum': '_read_enum',
        'TObject': '_read_object',

        'gws.core.types.Color': '_read_color',
        'gws.core.types.Crs': '_read_crs',
        'gws.core.types.Date': '_read_date',
        'gws.core.types.DateTime': '_read_datetime',
        'gws.core.types.DirPath': '_read_dirpath',
        'gws.core.types.Duration': '_read_duration',
        'gws.core.types.FilePath': '_read_filepath',
        'gws.core.types.FormatStr': '_read_formatstr',
        'gws.core.types.Regex': '_read_regex',
        'gws.core.types.Url': '_read_url',
    }

    def __init__(self, specs, path, strict, stack):
        self.specs = specs
        self.path = path
        self.strict = strict
        self.stack = stack
        self.push = stack.append if stack else lambda x: ...
        self.pop = stack.pop if stack else lambda: ...

    def read(self, val, type_name):
        if type_name in self._handlers:
            return getattr(self, self._handlers[type_name])(val, None)

        spec = self.specs.get(type_name)
        if not spec:
            raise Error('ERR_UNKNOWN', f'unknown type {type_name!r}', val)

        return getattr(self, self._handlers[spec['_']])(val, spec)

    # built-ins

    def _read_any(self, val, spec):
        return val

    def _read_bool(self, val, spec):
        if self.strict:
            return _ensure(val, bool)
        try:
            return bool(val)
        except:
            raise Error('ERR_MUST_BE_BOOL', 'must be true or false', val)

    def _read_bytes(self, val, spec):
        try:
            if isinstance(val, str):
                return val.encode('utf8', errors='strict')
            return bytes(val)
        except:
            raise Error('ERR_MUST_BE_BYTES', 'must be a byte buffer', val)

    def _read_float(self, val, spec):
        if self.strict:
            if isinstance(val, int):
                return float(val)
            return _ensure(val, float)
        try:
            return float(val)
        except:
            raise Error('ERR_MUST_BE_FLOAT', 'must be a float', val)

    def _read_int(self, val, spec):
        if self.strict:
            return _ensure(val, int)
        try:
            return int(val)
        except:
            raise Error('ERR_MUST_BE_INT', 'must be an integer', val)

    def _read_str(self, val, spec):
        if self.strict:
            return _ensure(val, str)
        try:
            return _to_string(val)
        except:
            raise Error('ERR_MUST_BE_STRING', 'must be a string', val)

    # anon complex types

    def _read_dict(self, val, spec):
        dct = {}
        for k, v in _ensure(val, dict).items():
            dct[k] = self.read(v, spec['value_t'])
        return dct

    def _read_list(self, val, spec):
        lst = self._read_any_list(val)
        res = []
        for n, v in enumerate(lst):
            self.push([v, n])
            res.append(self.read(v, spec['item_t']))
            self.pop()
        return res

    def _read_tuple(self, val, spec):
        lst = self._read_any_list(val)
        items = spec['items']

        if len(lst) != len(items):
            raise Error('ERR_BAD_TYPE', f"expected {_comma(items)}", val)

        res = []
        for n, v in enumerate(lst):
            self.push([v, n])
            res.append(self.read(v, items[n]))
            self.pop()
        return res

    def _read_any_list(self, val):
        if not self.strict and isinstance(val, str):
            val = val.strip()
            val = [v.strip() for v in val.split(',')] if val else []
        return _ensure(val, list)

    def _read_literal(self, val, spec):
        s = self._read_str(val, spec)
        if s not in spec['values']:
            raise Error('ERR_INVALID_CONST', f"expected {_comma(spec['values'])}, found {s!r}", val)
        return s

    def _read_optional(self, val, spec):
        if val is None:
            return val
        return self.read(val, spec['target_t'])

    def _read_union(self, val, spec):
        # @TODO no untyped unions yet
        raise Error('ERR_BAD_TYPE', 'not supported', val)

    def _read_variant(self, val, spec):
        val = _ensure(val, dict)
        if not self.strict:
            val = {k.lower(): v for k, v in val.items()}

        type_name = val.get('type')
        if not type_name:
            raise Error('ERR_MISSING_PROP', f"required property missing: 'type'", None)

        target_type = spec['members'].get(type_name)
        if not target_type:
            raise Error('ERR_BAD_TYPE', f"illegal type: {type_name!r}, expected {_comma(spec['members'])}", val)
        return self.read(val, target_type)

    # named types

    def _read_alias(self, val, spec):
        return self.read(val, spec['target_t'])

    def _read_enum(self, val, spec):
        # NB: our Enums accept both names (for configs) and values (for api calls)
        # this prevents silly things like Enum{foo=bar bar=123} but we don't care

        for k, v in spec['values'].items():
            if val == k or val == v:
                return v
        raise Error('ERR_BAD_ENUM', f"invalid value, expected {_comma(spec['values'])}", val)

    def _read_object(self, val, spec):
        val = _ensure(val, dict)
        if not self.strict:
            val = {k.lower(): v for k, v in val.items()}

        props = spec['props']
        res = {}

        for prop_name, prop_key in props.items():
            prop_val = val.get(prop_name if self.strict else prop_name.lower())
            self.push([prop_val, prop_name])
            res[prop_name] = self._read_property(prop_val, prop_key)
            self.pop()

        if self.strict:
            unknown = [k for k in val if k not in props]
            if unknown:
                raise Error('ERR_UNKNOWN_PROP', f"unknown keys: {_comma(unknown)}, expected: {_comma(props)}", val)

        return Data(res)

    def _read_property(self, val, key):
        prop_spec = self.specs[key]

        if val is not None:
            return self.read(val, prop_spec['property_t'])

        if not prop_spec['has_default']:
            raise Error('ERR_MISSING_PROP', f"required property missing: {prop_spec['ident']!r}", None)

        if prop_spec['default'] is None:
            return None

        # the default, if given, must match the type
        # NB, for Data objects, default={} will create an object with defaults
        return self.read(prop_spec['default'], prop_spec['property_t'])

    # special types

    def _read_color(self, val, spec):
        # @TODO: color validation
        return self._read_str(val, spec)

    def _read_crs(self, val, spec):
        # @TODO: crs validation
        return self._read_str(val, spec)

    def _read_date(self, val, spec):
        d = gws.lib.date.from_iso(str(val))
        if not d:
            raise Error('ERR_INVALID_DATE', 'invalid date', val)
        return gws.lib.date.to_iso_date(d)

    def _read_datetime(self, val, spec):
        d = gws.lib.date.from_iso(str(val))
        if not d:
            raise Error('ERR_INVALID_DATE', 'invalid date', val)
        return gws.lib.date.to_iso(d)

    def _read_dirpath(self, val, spec):
        path = gws.lib.os2.abs_path(val, self.path)
        if not gws.is_dir(path):
            raise Error('ERR_DIR_NOT_FOUND', 'directory not found', path)
        return path

    def _read_duration(self, val, spec):
        try:
            return gws.lib.units.parse_duration(val)
        except ValueError:
            raise Error('ERR_BAD_DURATION', 'invalid duration', val)

    def _read_filepath(self, val, spec):
        path = gws.lib.os2.abs_path(val, self.path)
        if not gws.is_file(path):
            raise Error('ERR_FILE_NOT_FOUND', 'file not found', path)
        return path

    def _read_formatstr(self, val, spec):
        # @TODO
        return self._read_str(val, spec)

    def _read_regex(self, val, spec):
        try:
            re.compile(val)
            return val
        except re.error as e:
            raise Error('ERR_BAD_REGEX', f"invalid regular expression: {e!r}", val)

    def _read_url(self, val, spec):
        # @TODO: url validation
        return self._read_str(val, spec)


# utils


def _ensure(val, klass):
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
    return ', '.join(sorted(str(x) for x in ls))


def _add_error_details(e: Error, path: str, stack):
    details = {}

    try:
        details['formatted_value'] = 'VALUE: ' + _format_value(e.args[2])
    except:
        details['formatted_value'] = ''

    details['path'] = 'PATH: ' + path

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
