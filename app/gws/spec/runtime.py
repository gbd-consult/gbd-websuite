"""Validate values according to specs"""

import re
import sys

import gws
import gws.lib.jsonx
import gws.lib.importer
import gws.types as t

from . import core, reader
from .generator import generator

Error = core.Error
ReadError = core.ReadError
GeneratorError = core.GeneratorError
LoadError = core.LoadError


def create(manifest_path: str = None, read_cache=False, write_cache=False) -> 'Object':
    cache_path = gws.TMP_DIR + '/spec_' + gws.to_uid(manifest_path or '') + '.json'

    if read_cache and gws.is_file(cache_path):
        try:
            gs = gws.lib.jsonx.from_path(cache_path)
            gws.log.debug(f'spec.create: loaded from {cache_path!r}')
            return Object(gs)
        except gws.lib.jsonx.Error:
            gws.log.exception(f'spec.create: load failed')

    ts = gws.time_start('SPEC GENERATOR')
    gs = generator.generate_specs(manifest_path=manifest_path)
    gws.time_end(ts)

    if write_cache:
        try:
            gws.lib.jsonx.to_path(cache_path, gs, pretty=True)
            gws.log.debug(f'spec.create: stored to {cache_path!r}')
        except gws.lib.jsonx.Error:
            gws.log.exception(f'spec.create: store failed')

    return Object(gs)


##


class Object(gws.ISpecRuntime):
    def __init__(self, gs):
        meta = gs['meta']
        self.manifest = gws.ApplicationManifest(meta['manifest'])
        self.manifestPath = meta['manifestPath']
        self.version = meta['version']

        self.specs = [core.Type(**s) for s in gs['specs']]

        self.index = {}
        for s in self.specs:
            self.index[s.uid] = s
            if s.extName:
                self.index[s.extName] = s

        self.strings = gs['strings']
        self.chunks = gs['chunks']

        self._descCache = {}

    def get_type(self, key):
        return self.index.get(key)

    def read(self, value, type_name, path='', options=None):
        r = reader.Reader(self, path, options)
        return r.read(value, type_name)

    def object_descriptor(self, name):
        if name in self._descCache:
            return self._descCache[name]

        typ = self.get_type(name)
        if not typ:
            return

        self._descCache[name] = gws.ExtObjectDescriptor(
            extName=typ.extName,
            ident=typ.ident,
            modName=typ.modName,
            modPath=typ.modPath,
            classPtr=None
        )

        return self._descCache[name]

    def get_class(self, classref, ext_type=None):
        cls, name, ext_name = self.parse_classref(classref)

        if cls:
            return cls

        if ext_name:
            name = ext_name + '.' + (ext_type or core.DEFAULT_VARIANT_TAG)

        desc = self.object_descriptor(name)
        if not desc:
            return

        if not desc.classPtr:
            if desc.modName in sys.modules:
                mod = sys.modules[desc.modName]
            else:
                try:
                    mod = gws.lib.importer.import_from_path(desc.modPath, gws.APP_DIR)
                except gws.lib.importer.Error as exc:
                    raise LoadError(f'cannot load class {classref!r} from {desc.modPath!r}') from exc
            desc.classPtr = getattr(mod, desc.ident)

        return desc.classPtr

    def command_descriptor(self, command_category, command_name):
        name = core.EXT_COMMAND_PREFIX + command_category + '.' + command_name

        if name in self._descCache:
            return self._descCache[name]

        typ = self.get_type(name)

        if not typ:
            return

        return gws.ExtCommandDescriptor(
            extName=typ.extName,
            tArg=typ.tArg,
            tOwner=typ.tOwner,
            owner=self.object_descriptor(typ.tOwner),
            methodName=typ.ident,
        )

    def cli_commands(self, lang='en'):
        strings = self.strings.get(lang) or self.strings['en']
        cmds = []

        for typ in self.specs:
            if not typ.extName.startswith(core.EXT_COMMAND_CLI_PREFIX):
                continue

            # serverStart -> [server, start]
            m = re.search(r'\.([a-z]+)(\w+)$', typ.extName)
            cmd1 = m.group(1)
            cmd2 = m.group(2).lower()

            entry = gws.Data(
                cmd1=cmd1,
                cmd2=cmd2,
                doc=strings.get(typ.uid) or self.strings['en'].get(typ.uid) or '',
            )
            cmds.append(entry)

            args = []
            arg_typ = self.get_type(typ.tArg)
            if arg_typ:
                for name, prop_type_uid in arg_typ.tProperties.items():
                    prop_typ = self.get_type(prop_type_uid)
                    args.append(gws.Data(
                        name=name,
                        doc=strings.get(prop_type_uid) or self.strings['en'].get(prop_type_uid) or '',
                        default=prop_typ.default,
                        hasDefault=prop_typ.hasDefault,
                    ))
            entry.args = sorted(args, key=lambda a: a.name)

        return sorted(cmds, key=lambda c: (c.cmd1, c.cmd2))

    def bundle_paths(self, category):
        if category == 'vendor':
            return [gws.APP_DIR + '/' + gws.JS_VENDOR_BUNDLE]
        if category == 'util':
            return [gws.APP_DIR + '/' + gws.JS_UTIL_BUNDLE]
        if category == 'app':
            paths = []
            for chunk in self.chunks:
                path = chunk['bundleDir'] + '/' + gws.JS_BUNDLE
                if gws.is_file(path) and path not in paths:
                    paths.append(path)
            return paths

    def parse_classref(self, classref: gws.ClassRef) -> t.Tuple[t.Optional[type], str, str]:
        if isinstance(classref, str):
            if gws.ext.is_name(classref):
                return None, '', classref
            return None, classref, ''
        name = gws.ext.name(classref)
        if name:
            return None, '', name
        if isinstance(classref, type):
            return classref, '', ''
        raise Error(f'invalid class reference {classref!r}')
    ##

    # def ext_type_list(self, category):
    #     return [
    #         spec.get('ext_type')
    #         for spec in self.specs.values()
    #         if spec.get('ext_category') == category
    #     ]
    #
    # def parse_command(self, cmd_name, cmd_method, params, with_strict_mode=True):
    #     name = cmd_method + '.' + cmd_name
    #     if name not in self.specs:
    #         return None
    #
    #     cmd_spec = self.specs[name]
    #     p = gws.Request(self.read_value(params, cmd_spec['arg'], '', with_strict_mode, with_error_details=False))
    #
    #     return gws.ExtCommandDescriptor(
    #         class_name=cmd_spec['class_name'],
    #         cmd_action=cmd_spec['cmd_action'],
    #         cmd_name=cmd_spec['cmd_name'],
    #         function_name=cmd_spec['function_name'],
    #         params=p or gws.Request(),
    #     )
    #
    # def is_a(self, class_name, class_name_part):
    #     for cnames in self.isa_map.values():
    #         if class_name in cnames and class_name_part in cnames:
    #             return True
    #
    # def real_class_names(self, class_name):
    #     return [
    #         name
    #         for name, cnames in self.isa_map.items()
    #         if class_name in cnames
    #     ]
