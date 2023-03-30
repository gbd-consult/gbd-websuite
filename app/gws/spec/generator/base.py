import os
import sys

from . import util

from gws.spec.core import *

ATOMS = ['any', 'bool', 'bytes', 'float', 'int', 'str']

BUILTINS = ATOMS + ['type', 'object', 'Exception', 'dict', 'list', 'set', 'tuple']

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
    APP_NAME + '.core.const',
    APP_NAME + '.core.data',
    APP_NAME + '.core.tree',
    APP_NAME + '.core.types',
    APP_NAME + '.core.util',
    APP_NAME + '.core.error',
    APP_NAME + '.core'
]

DEFAULT_EXT_SUPERS = {
    'config': APP_NAME + '.core.types.ConfigWithAccess',
    'props': APP_NAME + '.core.types.Props',
}

# prefix for gws.plugin class names
PLUGIN_PREFIX = APP_NAME + '.plugin'

# comment prefix for Type aliases
TYPE_COMMENT_PREFIX = 'type:'

# comment prefix for Variant aliases
VARIANT_COMMENT_PREFIX = 'variant:'

# inline comment symbol
INLINE_COMMENT_SYMBOL = '#:'

# where we are
SELF_DIR = os.path.dirname(__file__)

# path to `/repository-root/app`
APP_DIR = os.path.abspath(SELF_DIR + '/../../..')

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
    [APP_NAME + '', '/js/src/gws'],
    [APP_NAME + '.core', '/gws/core'],
    [APP_NAME + '.base', '/gws/base'],
    [APP_NAME + '.gis', '/gws/gis'],
    [APP_NAME + '.lib', '/gws/lib'],
    [APP_NAME + '.server', '/gws/server'],

]


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
    levels = 'ERROR', 'WARNING', 'INFO', 'DEBUG'

    def set_level(self, level):
        self.level = level

    def log(self, level, *args):
        if self.levels.index(level) <= self.levels.index(self.level):
            msg = f'[spec] {level}: ' + ' '.join(str(a) for a in args)
            sys.stdout.write(msg + '\n')
            sys.stdout.flush()

    def error(self, *args): self.log('ERROR', *args)

    def warn(self, *args): self.log('WARNING', *args)

    def info(self, *args): self.log('INFO', *args)

    def debug(self, *args): self.log('DEBUG', *args)


log = _Logger()


class Generator(Data):
    meta: dict
    types: Dict[str, Type]
    extRefs: Dict[str, List]
    specs = {}
    typescript = ''
    strings = {}

    rootDir = ''
    selfDir = ''
    outDir = ''
    manifestPath = ''

    debug = False

    chunks: List[dict]

    def __init__(self):
        super().__init__()
        self.aliases = {}
        self.types = {}
        self.extRefs = {}

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
        return comma.join(repr(v) for v in d['literalValues'])
    if c == C.OPTIONAL:
        return d['tTarget']
    if c == C.TUPLE:
        return comma.join(d['tItems'])
    if c == C.UNION:
        return comma.join(sorted(d['tItems']))
    if c == C.EXT:
        return d['extName']
    if c == C.VARIANT:
        if 'tMembers' in d:
            return comma.join(sorted(d['tMembers'].values()))
        return comma.join(sorted(d['tItems']))

    return ''
