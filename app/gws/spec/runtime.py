"""Validate values according to specs"""

import re
import sys

import gws
import gws.lib.json2
import gws.types as t

from . import core, reader
from .generator import generator

Error = core.Error
ReadError = core.ReadError
GeneratorError = core.GeneratorError

DEFAULT_TYPE = core.DEFAULT_TYPE


def create(manifest_path: str = None, read_cache=False, write_cache=False) -> 'Object':
    cache_path = gws.TMP_DIR + '/spec_' + gws.to_uid(manifest_path or '') + '.json'

    if read_cache and gws.is_file(cache_path):
        try:
            gs = gws.lib.json2.from_path(cache_path)
            gws.log.debug(f'spec.create: loaded from {cache_path!r}')
            return Object(gs)
        except gws.lib.json2.Error:
            gws.log.exception(f'spec.create: load failed')

    ts = gws.time_start('SPEC GENERATOR')
    gs = generator.generate_specs(manifest_path=manifest_path)
    gws.time_end(ts)

    if write_cache:
        try:
            gws.lib.json2.to_path(cache_path, gs, pretty=True)
            gws.log.debug(f'spec.create: stored to {cache_path!r}')
        except gws.lib.json2.Error:
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

    def get(self, key):
        return self.index.get(key)

    def read(self, value, type_name, path='', strict_mode=True, verbose_errors=True, accept_extra_props=False):
        r = reader.Reader(self, path, strict_mode, verbose_errors, accept_extra_props)
        return r.read(value, type_name)

    def object_descriptor(self, name):
        typ = self.get(name)
        if not typ:
            return
        return gws.ExtObjectDescriptor(
            extName=typ.extName,
            ident=typ.ident,
            modName=typ.modName,
            modPath=typ.modPath,
            classPtr=None
        )

    def command_descriptor(self, command_category, command_name):
        name = core.EXT_COMMAND_PREFIX + command_category + '.' + command_name
        typ = self.get(name)

        if not typ:
            return

        return gws.ExtCommandDescriptor(
            extName=typ.extName,
            tArg=typ.tArg,
            tOwner=typ.tOwner,
            methodName=typ.ident,
        )

    def cli_docs(self, lang='en'):
        strings = self.strings.get(lang) or self.strings['en']
        docs = []

        for typ in self.specs:
            if not typ.extName.startswith(core.EXT_COMMAND_CLI_PREFIX):
                continue

            # serverStart -> [server, start]
            m = re.search(r'\.([a-z]+)(\w+)$', typ.extName)
            cmd1 = m.group(1)
            cmd2 = m.group(2).lower()

            tab = ' ' * 4
            text = [f'gws {cmd1} {cmd2}']

            doc = strings.get(typ.uid) or self.strings['en'].get(typ.uid) or ''
            if doc:
                text.append(f'{tab} {doc}')

            arg_typ = self.get(typ.tArg)
            if arg_typ:
                for name, prop_type_uid in sorted(arg_typ.tProperties.items()):
                    prop_typ = self.get(prop_type_uid)
                    doc = strings.get(prop_type_uid) or self.strings['en'].get(prop_type_uid) or ''
                    line = f'{tab}{tab}--{name:15} {doc}'
                    if not prop_typ.hasDefault:
                        line += ' (*)'
                    elif prop_typ.default:
                        line += f' (default {prop_typ.default!r})'
                    text.append(line)

            text.append('')

            docs.append([cmd1, cmd2, '\n'.join(text)])

        return sorted(docs)

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
