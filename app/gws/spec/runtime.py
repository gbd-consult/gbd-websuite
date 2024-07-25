"""Validate values according to specs"""

from typing import Optional

import re
import sys

import gws
import gws.lib.jsonx
import gws.lib.importer

from . import core, reader
from .generator import generator

Error = core.Error
ReadError = core.ReadError
GeneratorError = core.GeneratorError
LoadError = core.LoadError


def create(manifest_path: str = None, read_cache=False, write_cache=False) -> 'Object':
    gs = get_spec(manifest_path, read_cache, write_cache)
    return Object(gs)


def get_spec(manifest_path: str = None, read_cache=False, write_cache=False) -> dict:
    cache_path = gws.u.ensure_dir(gws.c.SPEC_DIR) + '/spec_' + gws.u.to_uid(manifest_path or '') + '.json'

    if read_cache and gws.u.is_file(cache_path):
        try:
            gs = gws.lib.jsonx.from_path(cache_path)
            gws.log.debug(f'spec.create: loaded from {cache_path!r}')
            return gs
        except gws.lib.jsonx.Error:
            gws.log.exception(f'spec.create: load failed')

    if manifest_path:
        gws.log.debug(f'spec.create: using manifest {manifest_path!r}...')

    gws.debug.time_start('SPEC GENERATOR')
    gs = generator.generate_specs(manifest_path=manifest_path)
    gws.debug.time_end()

    if write_cache:
        try:
            gws.lib.jsonx.to_path(cache_path, gs, pretty=True)
            gws.log.debug(f'spec.create: stored to {cache_path!r}')
        except gws.lib.jsonx.Error:
            gws.log.exception(f'spec.create: store failed')

    return gs


##


class Object(gws.SpecRuntime):
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

        self.appBundlePaths = []
        for chunk in self.chunks:
            path = chunk['bundleDir'] + '/' + gws.c.JS_BUNDLE
            if path not in self.appBundlePaths:
                self.appBundlePaths.append(path)

        self._descCache = {}

    def __getstate__(self):
        self._descCache = {}
        return vars(self)

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
            extType=typ.extName.split('.').pop(),
            ident=typ.ident,
            modName=typ.modName,
            modPath=typ.modPath,
            classPtr=None
        )

        return self._descCache[name]

    def register_object(self, classref, ext_type, cls):
        _, _, ext_name = self.parse_classref(classref)
        if not ext_name:
            raise Error(f'invalid class reference {classref!r}')
        ext_name += '.' + ext_type
        setattr(cls, 'extName', ext_name)
        setattr(cls, 'extType', ext_type)
        self._descCache[ext_name] = gws.ExtObjectDescriptor(
            extName=ext_name,
            extType=ext_type,
            ident=cls.__name__,
            modName='',
            modPath='',
            classPtr=cls
        )

    def get_class(self, classref, ext_type=None):
        cls, real_name, ext_name = self.parse_classref(classref)

        if cls:
            return cls

        if real_name:
            desc = self.object_descriptor(real_name)
        elif ext_name:
            desc = self.object_descriptor(ext_name + '.' + (ext_type or core.DEFAULT_VARIANT_TAG))
        else:
            desc = None

        if not desc:
            return

        if not desc.classPtr:
            try:
                mod = gws.lib.importer.import_from_path(desc.modPath, gws.c.APP_DIR)
            except gws.lib.importer.Error as exc:
                raise LoadError(f'cannot load class {classref!r} from {desc.modPath!r}') from exc
            desc.classPtr = getattr(mod, desc.ident)
            setattr(desc.classPtr, 'extName', desc.extName)
            setattr(desc.classPtr, 'extType', desc.extType)

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
            extType=typ.extName.split('.').pop(),
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
                        type=prop_typ.tValue,
                        doc=strings.get(prop_type_uid) or self.strings['en'].get(prop_type_uid) or '',
                        defaultValue=prop_typ.defaultValue,
                        hasDefault=prop_typ.hasDefault,
                    ))
            entry.args = sorted(args, key=lambda a: a.name)

        return sorted(cmds, key=lambda c: (c.cmd1, c.cmd2))

    def parse_classref(self, classref: gws.ClassRef) -> tuple[Optional[type], str, str]:
        ext_name = gws.ext.name_for(classref)
        if ext_name:
            return None, '', ext_name

        if isinstance(classref, str):
            return None, classref, ''

        if isinstance(classref, type):
            return classref, '', ''

        raise Error(f'invalid class reference {classref!r}')
