"""Generator state, a data class and the generator logger."""

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
    """Simple data object. Missing attributes return ``None``."""

    def __init__(self, **kwargs):
        """Create an object with the given attributes.

        Args:
            **kwargs: Attribute values.
        """

        vars(self).update(kwargs)

    def __repr__(self):
        return repr(vars(self))

    def get(self, k, default=None):
        """Get an attribute value.

        Args:
            k: Attribute name.
            default: Value to return if the attribute is not set.

        Returns:
            The attribute value or the default.
        """

        return vars(self).get(k, default)

    def __getattr__(self, item):
        """Return ``None`` for attributes that are not set."""

        return None


class _Logger:
    """Logger that writes messages to stdout."""

    level = 'INFO'
    levels = 'ERROR', 'WARNING', 'INFO', 'DEBUG'

    def set_level(self, level):
        """Set the log level.

        Args:
            level: One of ``ERROR``, ``WARNING``, ``INFO``, ``DEBUG``.
        """

        self.level = level

    def log(self, level, *args):
        """Write a message if the level is enabled.

        Args:
            level: Message level.
            *args: Message parts, joined with spaces.
        """

        if self.levels.index(level) <= self.levels.index(self.level):
            msg = f'[spec] {level}: ' + ' '.join(str(a) for a in args)
            sys.stdout.write(msg + '\n')
            sys.stdout.flush()

    def error(self, *args):
        """Log an error message.

        Args:
            *args: Message parts.
        """

        self.log('ERROR', *args)

    def warning(self, *args):
        """Log a warning message.

        Args:
            *args: Message parts.
        """

        self.log('WARNING', *args)

    def info(self, *args):
        """Log an info message.

        Args:
            *args: Message parts.
        """

        self.log('INFO', *args)

    def debug(self, *args):
        """Log a debug message.

        Args:
            *args: Message parts.
        """

        self.log('DEBUG', *args)


log = _Logger()


class Generator:
    """Generator state, shared by all generator steps.

    Attributes:
        aliases: Maps alias names (from imports and global names) to target names.
        chunks: Source code chunks.
        meta: Build-time metadata.
        typeDict: All types, keyed by uid.
        serverTypes: Types extracted for the server.
        specData: The resulting spec data.
        configRef: Configuration references in Markdown, keyed by language.
        strings: Documentation strings, keyed by language and type uid.
        manifestPath: Path to the application manifest.
        outDir: Output directory.
        rootDir: Application root directory (``app``).
        selfDir: Directory of the spec package.
        typescript: Generated TypeScript API.
        debug: If True, dump the state after each step.
    """

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
        """Create a type and add it to the type dictionary.

        Named types use the name as uid. Other types get an automatic uid made
        of the type kind and the uids of the types they refer to, so that equal
        types share the uid.

        Args:
            **kwargs: Type attributes.

        Returns:
            The new type.

        Raises:
            ``GeneratorError``: If no automatic uid can be created for the type kind.
        """

        if kwargs.get('name'):
            kwargs['uid'] = kwargs['name']
        if not kwargs.get('uid'):
            kwargs['uid'] = kwargs['c'] + ':' + _auto_uid(kwargs)
        typ = core.make_type(kwargs)
        self.typeDict[typ.uid] = typ
        return typ

    def get_type(self, uid) -> Optional[Type]:
        """Get a type by uid.

        Args:
            uid: Type uid.

        Returns:
            The type or ``None`` if not found.
        """

        return self.typeDict.get(uid)

    def require_type(self, uid) -> Type:
        """Get a type by uid, failing if it does not exist.

        Args:
            uid: Type uid.

        Returns:
            The type.

        Raises:
            ``GeneratorError``: If the type is not found.
        """

        typ = self.typeDict.get(uid)
        if not typ:
            raise GeneratorError(f'unknown type {uid!r}')
        return typ

    def dump(self, tag):
        """Write the generator state to ``<outDir>/<tag>.debug.json`` in debug mode.

        Args:
            tag: File name tag.
        """

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
