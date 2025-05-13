from typing import Optional
import sys

from .. import core
from . import util

c = core.c
v = core.v
Error = core.Error
GeneratorError = core.GeneratorError
LoadError = core.LoadError
ReadError = core.ReadError
Type = core.Type


class Data:
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

    def error(self, *args):
        self.log('ERROR', *args)

    def warning(self, *args):
        self.log('WARNING', *args)

    def info(self, *args):
        self.log('INFO', *args)

    def debug(self, *args):
        self.log('DEBUG', *args)


log = _Logger()


class Generator:
    def __init__(self):
        self.aliases: dict[str, str] = {}
        self.chunks: list[core.Chunk] = []
        self.meta: dict = {}
        self.typeDict: dict[str, Type] = {}
        self.serverTypes: list[Type] = []
        self.specData: core.SpecData
        self.configRef = {}
        self.strings = {}
        self.manifestPath = ''
        self.outDir = ''
        self.rootDir = ''
        self.selfDir = ''
        self.typescript = ''
        self.debug = False

    def add_type(self, **kwargs):
        if kwargs.get('name'):
            kwargs['uid'] = kwargs['name']
        if not kwargs.get('uid'):
            kwargs['uid'] = kwargs['c'] + ':' + _auto_uid(kwargs)
        typ = core.make_type(kwargs)
        self.typeDict[typ.uid] = typ
        return typ

    def get_type(self, uid) -> Optional[Type]:
        return self.typeDict.get(uid)

    def require_type(self, uid) -> Type:
        typ = self.typeDict.get(uid)
        if not typ:
            raise GeneratorError(f'unknown type {uid!r}')
        return typ

    def dump(self, tag):
        if self.debug:
            util.write_json(self.outDir + '/' + tag + '.debug.json', vars(self))


def _auto_uid(args):
    tc = args['c']
    if tc == c.DICT:
        return args['tKey'] + ',' + args['tValue']
    if tc == c.LIST:
        return args['tItem']
    if tc == c.SET:
        return args['tItem']
    if tc == c.LITERAL:
        return _comma(repr(v) for v in args['literalValues'])
    if tc == c.OPTIONAL:
        return args['tTarget']
    if tc == c.TUPLE:
        return _comma(args['tItems'])
    if tc == c.UNION:
        return _comma(sorted(args['tItems']))
    if tc == c.CALLABLE:
        return _comma(sorted(args['tItems']))
    if tc == c.EXT:
        return args['extName']
    if tc == c.VARIANT:
        if 'tMembers' in args:
            return _comma(sorted(args['tMembers'].values()))
        return _comma(sorted(args['tItems']))
    raise GeneratorError(f'auto uid for {tc!r} not implemented: {args}')


_comma = ','.join
