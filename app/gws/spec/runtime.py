"""Validate values according to specs"""

import re

import gws
import gws.gis.crs
import gws.lib.date
import gws.lib.json2
import gws.lib.os2
import gws.lib.units

from . import core, generator


class Error(Exception):
    pass


def create(manifest_path: str = None) -> 'Object':
    gs = _generate(manifest_path)
    return Object(gs)


def create_and_store(manifest_path: str = None) -> 'Object':
    gs = _generate(manifest_path)
    cc = _cache_path(manifest_path)

    try:
        gws.lib.json2.to_path(cc, gs, pretty=True)
        gws.log.debug(f'spec.create: stored to {cc!r}')
    except:
        gws.log.exception(f'spec.create: store failed')

    return Object(gs)


def load(manifest_path: str = None) -> 'Object':
    cc = _cache_path(manifest_path)

    if gws.is_file(cc):
        try:
            gs = gws.lib.json2.from_path(cc)
            gws.log.debug(f'spec.load: loaded from {cc!r}')
            return Object(gs)
        except:
            gws.log.exception(f'spec.load: load failed')

    return create_and_store(manifest_path)


def _generate(manifest_path):
    ts = gws.time_start('SPEC GENERATOR')
    try:
        gs = generator.generate(manifest_path)



        return {
            'meta': gws.SpecMeta
        }
    except Exception as exc:
        raise Error(f'system error, spec generator failed') from exc
    gws.time_end(ts)
    return gs


def _cache_path(manifest_path):
    return gws.TMP_DIR + '/spec_' + gws.sha256(manifest_path or '') + '.json'


##


class Object(gws.ISpecRuntime):
    def __init__(self, gs):
        self.meta = core.Meta(gs['meta'])
        self.manifest = gws.Manifest(self.meta.manifest)
        self.specs = [gws.Data(s) for s in gs['specs']]
        self.strings = gs['strings']

    def bundle_paths(self, category):
        if category == 'vendor':
            return [self.meta['VENDOR_BUNDLE_PATH']]
        if category == 'util':
            return [self.meta['UTIL_BUNDLE_PATH']]
        if category == 'app':
            paths = []
            for chunk in self.meta['chunks']:
                path = chunk['bundleDir'] + '/' + self.meta['BUNDLE_FILENAME']
                if gws.is_file(path) and path not in paths:
                    paths.append(path)
            return paths

    def ext_type_list(self, category):
        return [
            spec.get('ext_type')
            for spec in self.specs.values()
            if spec.get('ext_category') == category
        ]

    def parse_command(self, cmd_name, cmd_method, params, with_strict_mode=True):
        name = cmd_method + '.' + cmd_name
        if name not in self.specs:
            return None

        cmd_spec = self.specs[name]
        p = gws.Params(self.read_value(params, cmd_spec['arg'], '', with_strict_mode, with_error_details=False))

        return gws.ExtCommandDescriptor(
            class_name=cmd_spec['class_name'],
            cmd_action=cmd_spec['cmd_action'],
            cmd_name=cmd_spec['cmd_name'],
            function_name=cmd_spec['function_name'],
            params=p or gws.Params(),
        )

    def is_a(self, class_name, class_name_part):
        for cnames in self.isa_map.values():
            if class_name in cnames and class_name_part in cnames:
                return True

    def real_class_names(self, class_name):
        return [
            name
            for name, cnames in self.isa_map.items()
            if class_name in cnames
        ]

    def object_descriptor(self, class_name):
        s = self.specs.get(class_name)
        return gws.ExtObjectDescriptor(s) if s else None

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

    def read_value(self, value, type_name, path='', with_strict_mode=True, with_error_details=True, with_internal_objects=False):
        stack = None

        if with_error_details:
            stack = [[value, type_name]]

        rd = _Reader(self.specs, path, stack, with_strict_mode, with_error_details, with_internal_objects)

        if not with_error_details:
            return rd.read(value, type_name)

        try:
            return rd.read(value, type_name)
        except Error as e:
            raise _add_error_details(e, path, stack)


##

_READERS = {
    'any': '_read_any',
    'bool': '_read_bool',
    'bytes': '_read_bytes',
    'float': '_read_float',
    'int': '_read_int',
    'str': '_read_str',

    'TDict': '_read_dict',
    'TList': '_read_list',
    'TSet': '_read_set',
    'TLiteral': '_read_literal',
    'TOptional': '_read_optional',
    'TTuple': '_read_tuple',
    'TUnion': '_read_union',
    'TVariant': '_read_variant',

    'TAlias': '_read_alias',
    'TEnum': '_read_enum',
    'TNode': '_read_object',
    'TRecord': '_read_object',

    'gws.core.types.Color': '_read_color',
    'gws.core.types.CrsId': '_read_crs',
    'gws.core.types.Date': '_read_date',
    'gws.core.types.DateTime': '_read_datetime',
    'gws.core.types.DirPath': '_read_dirpath',
    'gws.core.types.Duration': '_read_duration',
    'gws.core.types.FilePath': '_read_filepath',
    'gws.core.types.FormatStr': '_read_formatstr',
    'gws.core.types.Regex': '_read_regex',
    'gws.core.types.Url': '_read_url',
}


class _Reader:
    def __init__(self, specs, path, stack, with_strict_mode, with_error_details, with_internal_objects):
        self.specs = specs
        self.path = path
        self.stack = stack
        self.push = stack.append if stack else lambda x: ...
        self.pop = stack.pop if stack else lambda: ...
        self.with_strict_mode = with_strict_mode
        self.with_error_details = with_error_details
        self.with_internal_objects = with_internal_objects

    def read(self, val, type_name):
        if type_name in _READERS:
            return getattr(self, _READERS[type_name])(val, None)

        spec = self.specs.get(type_name)
        if not spec:
            raise Error('ERR_UNKNOWN', f'unknown type {type_name!r}', val)

        return getattr(self, _READERS[spec['_']])(val, spec)

    # built-ins

    def _read_any(self, val, spec):
        return val

    def _read_bool(self, val, spec):
        if self.with_strict_mode:
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
        if self.with_strict_mode:
            if isinstance(val, int):
                return float(val)
            return _ensure(val, float)
        try:
            return float(val)
        except:
            raise Error('ERR_MUST_BE_FLOAT', 'must be a float', val)

    def _read_int(self, val, spec):
        if self.with_strict_mode:
            return _ensure(val, int)
        try:
            return int(val)
        except:
            raise Error('ERR_MUST_BE_INT', 'must be an integer', val)

    def _read_str(self, val, spec):
        if self.with_strict_mode:
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

    def _read_set(self, val, spec):
        lst = self._read_list(val, spec)
        return set(lst)

    def _read_tuple(self, val, spec):
        lst = self._read_any_list(val)
        items = spec['items']

        if len(lst) != len(items):
            raise Error('ERR_BAD_TYPE', f"expected: {_comma(items)}", val)

        res = []
        for n, v in enumerate(lst):
            self.push([v, n])
            res.append(self.read(v, items[n]))
            self.pop()
        return res

    def _read_any_list(self, val):
        if not self.with_strict_mode and isinstance(val, str):
            val = val.strip()
            val = [v.strip() for v in val.split(',')] if val else []
        return _ensure(val, list)

    def _read_literal(self, val, spec):
        s = self._read_str(val, spec)
        if s not in spec['values']:
            raise Error('ERR_INVALID_CONST', f"invalid value: {s!r}, expected: {_comma(spec['values'])}", val)
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
        if not self.with_strict_mode:
            val = {k.lower(): v for k, v in val.items()}

        type_name = val.get('type')
        if not type_name:
            raise Error('ERR_MISSING_PROP', f"required property missing: 'type'", None)

        target_type = spec['members'].get(type_name)
        if not target_type:
            raise Error('ERR_BAD_TYPE', f"illegal type: {type_name!r}, expected: {_comma(spec['members'])}", val)
        return self.read(val, target_type)

    # named types

    def _read_alias(self, val, spec):
        return self.read(val, spec['target_t'])

    def _read_enum(self, val, spec):
        # NB: our Enums accept both names (for configs) and values (for api calls)
        # this prevents silly things like Enum{foo=bar bar=123} but we don't care
        #
        # this reader returns a value, it's up to the caller to convert it to the actual enum

        for k, v in spec['values'].items():
            if val == k or val == v:
                return v
        raise Error('ERR_BAD_ENUM', f"invalid value: {val!r}, expected: {_comma(spec['values'])}", val)

    def _read_object(self, val, spec):
        val = _ensure(val, dict)
        if not self.with_strict_mode:
            val = {k.lower(): v for k, v in val.items()}

        props = spec['props']
        res = {}

        for prop_name, prop_key in props.items():
            prop_val = val.get(prop_name if self.with_strict_mode else prop_name.lower())
            self.push([prop_val, prop_name])
            res[prop_name] = self._read_property(prop_val, prop_key)
            self.pop()

        unknown = []

        for k in val:
            if k not in props:
                if self.with_internal_objects:
                    res[k] = val[k]
                elif self.with_strict_mode:
                    unknown.append(k)

        if unknown:
            raise Error('ERR_UNKNOWN_PROP', f"unknown keys: {_comma(unknown)}, expected: {_comma(props)}", val)

        return gws.Data(res)

    def _read_property(self, val, key):
        prop_spec = self.specs[key]

        if val is not None:
            return self.read(val, prop_spec['property_t'])

        if not prop_spec['has_default'] and not self.with_internal_objects:
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
        crs = gws.gis.crs.get(val)
        if not crs:
            raise Error('ERR_INVALID_CRS', f'invalid crs: {val!r}', val)
        return crs.srid

    def _read_date(self, val, spec):
        d = gws.lib.date.from_iso(str(val))
        if not d:
            raise Error('ERR_INVALID_DATE', f'invalid date: {val!r}', val)
        return gws.lib.date.to_iso_date(d)

    def _read_datetime(self, val, spec):
        d = gws.lib.date.from_iso(str(val))
        if not d:
            raise Error('ERR_INVALID_DATE', f'invalid date: {val!r}', val)
        return gws.lib.date.to_iso(d)

    def _read_dirpath(self, val, spec):
        path = gws.lib.os2.abs_path(val, self.path)
        if not gws.is_dir(path):
            raise Error('ERR_DIR_NOT_FOUND', f'directory not found: {path!r}', path)
        return path

    def _read_duration(self, val, spec):
        try:
            return gws.lib.units.parse_duration(val)
        except ValueError:
            raise Error('ERR_BAD_DURATION', f'invalid duration: {val!r}', val)

    def _read_filepath(self, val, spec):
        path = gws.lib.os2.abs_path(val, self.path)
        if not gws.is_file(path):
            raise Error('ERR_FILE_NOT_FOUND', f'file not found: {path!r}', path)
        return path

    def _read_formatstr(self, val, spec):
        # @TODO
        return self._read_str(val, spec)

    def _read_regex(self, val, spec):
        try:
            re.compile(val)
            return val
        except re.error as e:
            raise Error('ERR_BAD_REGEX', f'invalid regular expression: {val!r}: {e!r}', val)

    def _read_url(self, val, spec):
        # @TODO: url validation
        return self._read_str(val, spec)


# utils


def _ensure(val, klass):
    if isinstance(val, klass):
        return val
    if klass == list and isinstance(val, tuple):
        return list(val)
    if klass == dict and gws.is_data_object(val):
        return vars(val)
    raise Error('ERR_WRONG_TYPE', f"wrong type: {_classname(type(val))!r}, expected: {_classname(klass)!r}", val)


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
