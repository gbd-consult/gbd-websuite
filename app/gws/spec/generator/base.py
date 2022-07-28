import os

from typing import Any, Dict, List

from . import util

import gws.spec.core

C = gws.spec.core.C
Meta = gws.spec.core.Meta
Type = gws.spec.core.Type

ATOMS = ['any', 'bool', 'bytes', 'float', 'int', 'str']

BUILTINS = ATOMS + ['type', 'object', 'Exception', 'dict', 'list', 'tuple']

BUILTIN_TYPES = [
    'Any',
    'Callable',
    'Dict',
    'List',
    'Literal',
    'Optional',
    'Protocol',
    'Set',
    'Tuple',
    'Union',
    'Enum',
]

# those star-imported in gws/__init__.py
GLOBAL_MODULES = [
    'gws.core.const',
    'gws.core.data',
    'gws.core.tree',
    'gws.core.types',
    'gws.core.util',
    'gws.core.error',
    'gws.core'
]

DEFAULT_EXT_SUPERS = {
    'config': 'gws.core.types.ConfigWithAccess',
    'props': 'gws.core.types.Props',
}

# prefix for our decorators and ext classes
EXT_PREFIX = 'gws.ext'
EXT_CONFIG_PREFIX = EXT_PREFIX + '.config.'
EXT_PROPS_PREFIX = EXT_PREFIX + '.props.'
EXT_OBJECT_PREFIX = EXT_PREFIX + '.object.'
EXT_COMMAND_PREFIX = EXT_PREFIX + '.command.'
EXT_COMMAND_API_PREFIX = EXT_COMMAND_PREFIX + 'api.'

# prefix for gws.plugin class names
PLUGIN_PREFIX = 'gws.plugin'

# tag property name for Variant types
TAG_PROPERTY = 'type'

# comment prefix for Type aliases
TYPE_COMMENT_PREFIX = 'type:'

# comment prefix for Variant aliases
VARIANT_COMMENT_PREFIX = 'variant:'

# inline comment symbol
INLINE_COMMENT_SYMBOL = '#:'

# path to `/repository-root/app`
APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../../..')

EXCLUDE_PATHS = ['___', '/vendor/', 'test', 'core/ext']

FILE_KINDS = [
    ['.py', 'python'],
    ['/index.ts', 'ts'],
    ['/index.tsx', 'ts'],
    ['/index.css.js', 'css'],
    ['.theme.css.js', 'theme'],
    ['/strings.ini', 'strings'],
]

PLUGIN_DIR = '/gws/plugin'

SYSTEM_CHUNKS = [
    ['gws', '/js/src/gws'],
    ['gws.core', '/gws/core'],
    ['gws.base', '/gws/base'],
    ['gws.gis', '/gws/gis'],
    ['gws.lib', '/gws/lib'],
    ['gws.server', '/gws/server'],

]

OUT_DIR = '/build'


class Data:  # type: ignore
    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def __repr__(self):
        return repr(vars(self))

    def get(self, k, default=None):
        return vars(self).get(k, default)

    def __getattr__(self, item):
        return None


class _Logger:
    level = 'INFO'

    def set_level(self, level):
        self.level = level

    def log(self, level, *args):
        levels = 'ERROR', 'WARNING', 'INFO', 'DEBUG'
        if levels.index(level) <= levels.index(self.level):
            print(f'[spec] {level}:', *args)

    def error(self, *args): self.log('ERROR', *args)

    def warn(self, *args): self.log('WARNING', *args)

    def info(self, *args): self.log('INFO', *args)

    def debug(self, *args): self.log('DEBUG', *args)


log = _Logger()


class Chunk(Data):
    name: str
    sourceDir: str
    bundleDir: str


class Generator(Data):
    meta: Meta
    types: Dict[str, Type] = {}
    specs = {}
    typescript = ''
    strings = {}

    rootDir = ''
    outDir = ''
    manifestPath = ''

    debug = False

    chunks: List[Chunk]

    def __init__(self):
        super().__init__()
        self.aliases = {}
        self.types = {}

    def new_type(self, c, **kwargs):
        if kwargs.get('name'):
            t = Type(**kwargs)
            t.c = c
            t.uid = t.name
            return t

        uid = c + ',' + _auto_uid(c, kwargs)
        if uid in self.types:
            return self.types[uid]

        t = Type(**kwargs)
        t.c = c
        t.uid = uid
        return t

    def dump(self, tag):
        if self.debug:
            util.write_json(self.outDir + '/' + tag + '.debug.json', self)


def _auto_uid(c, d):
    comma = ','
    if c == C.DICT:
        return d['tKey'] + comma + d['tValue']
    if c == C.LIST:
        return d['tItem']
    if c == C.SET:
        return d['tItem']
    if c == C.LITERAL:
        return comma.join(repr(v) for v in d['values'])
    if c == C.OPTIONAL:
        return d['tTarget']
    if c == C.TUPLE:
        return comma.join(d['tItems'])
    if c == C.UNION:
        return comma.join(sorted(d['tItems']))
    if c == C.VARIANT:
        if 'tMembers' in d:
            return comma.join(sorted(d['tMembers'].values()))
        return comma.join(sorted(d['tItems']))

    return ''


class Error(Exception):
    pass
