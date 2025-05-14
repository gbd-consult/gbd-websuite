from typing import TypeAlias, Any
import os


class Error(Exception):
    pass


class GeneratorError(Error):
    pass


class ReadError(Error):
    pass


class LoadError(Error):
    pass


class c:
    """Type kinds."""

    ATOM = 'ATOM'
    """Atomic, one of the built-in types."""
    CLASS = 'CLASS'
    """Class, a user-defined type."""
    CALLABLE = 'CALLABLE'
    """Callable, a callable argument."""
    CONSTANT = 'CONSTANT'
    """Constant type."""
    DICT = 'DICT'
    """Generic dictionary type."""
    ENUM = 'ENUM'
    """Enum type."""
    EXPR = 'EXPR'
    """Compile-time expression type."""
    FUNCTION = 'FUNCTION'
    """Function, a callable type."""
    LIST = 'LIST'
    """Generic list type."""
    LITERAL = 'LITERAL'
    """Literal type."""
    METHOD = 'METHOD'
    """Method, a callable type with a specific signature."""
    MODULE = 'MODULE'
    """Module type."""
    NONE = 'NONE'
    """None type."""
    OPTIONAL = 'OPTIONAL'
    """Optional, a type that can be None."""
    PROPERTY = 'PROPERTY'
    """Property, a type that is a property of a class."""
    SET = 'SET'
    """Generic set type."""
    TUPLE = 'TUPLE'
    """Generic tuple type."""
    TYPE = 'TYPE'
    """Type alias."""
    UNION = 'UNION'
    """Union, a type that can be one of several types."""
    UNDEFINED = 'UNDEFINED'
    """Undefined, a type that is not defined."""
    VARIANT = 'VARIANT'
    """Variant, a type that can be one of several types with a tag."""

    EXT = 'EXT'
    """Extension, a ``gws.ext`` alias."""
    COMMAND = 'COMMAND'
    """Command, a method decorated as ``gws.ext.command``."""


TypeKind: TypeAlias = str
"""Type kind, one of the constants in `c`."""
TypeUid: TypeAlias = str
"""Type unique identifier, a string that identifies the type."""


class Type:
    """Type data structure, repsents a GWS type."""

    c: TypeKind
    """Type class, one of the constants in `c`."""
    uid: TypeUid
    """Type unique identifier, a string that identifies the type."""

    extName: str = ''
    """Name of the extension that defines this, if any."""

    doc: str = ''
    """Documentation string for the type."""
    ident: str = ''
    """Source code identifier for the, used in the documentation."""
    name: str = ''
    """Name of the type."""
    pos: str = ''
    """Source code position of the type definition."""

    modName: str = ''
    """Name of the module that defines this type."""
    modPath: str = ''
    """Path to the module that defines this type."""

    tArg: TypeUid = ''
    """For c.METHOD types, type uid of its last argument."""
    tItem: TypeUid = ''
    """For c.LIST, c.SET and c.DICT types, type uid of its item."""
    tKey: TypeUid = ''
    """For c.DICT types, type uid of its key."""
    tModule: TypeUid = ''
    """Type uid of the type's module."""
    tOwner: TypeUid = ''
    """For c.PROPERTY types, type uid of the type that owns this property."""
    tReturn: TypeUid = ''
    """For c.METHOD types, type uid of its return value."""
    tTarget: TypeUid = ''
    """For c.TYPE or c.EXT types, type uid of the target type."""
    tValue: TypeUid = ''
    """For c.PROPERTY types, type uid of the constant value."""

    tArgs: list[TypeUid] = []
    """For c.METHOD types, type uids of its arguments."""
    tItems: list[TypeUid] = []
    """For c.UNION or c.TUPLE types, type uids of its items."""
    tSupers: list[TypeUid] = []
    """For c.CLASS types, type uids of its super types."""
    tMembers: dict[str, TypeUid] = {}
    """For c.VARIANT types, type uids of its members."""
    tProperties: dict[str, TypeUid] = {}
    """For c.CLASS types, type uids of its properties."""

    defaultValue: Any = None
    """Default value for a property."""
    defaultExpression: Any = None
    """Default expression for a property."""
    hasDefault: bool = False
    """True if the type has a default value."""
    constValue: Any = None
    """Constant value for a constant type."""

    enumDocs: dict = {}
    """Documentation strings for the enum values."""
    enumValues: dict = {}
    """Enum values for the enum type."""

    literalValues: list = []
    """Literal values for the c.LITERAL type."""


def make_type(args: dict):
    typ = Type()
    vars(typ).update(args)
    return typ


class Chunk:
    """Source code chunk."""

    name: str
    """Name of the chunk."""
    sourceDir: str
    """Source directory of the chunk."""
    bundleDir: str
    """Directory to save the compiled chunk bundle."""
    paths: dict[str, list[str]]
    """Source code paths."""
    exclude: list[str]
    """List of patterns to exclude from the chunk."""


class SpecData:
    """Specs data structure."""

    meta: dict
    """Meta data for the specs."""
    chunks: list[Chunk]
    """List of chunks."""
    serverTypes: list[Type]
    """List of types used by the server (configuration types, request types and commands)."""
    strings: dict
    """Documentation strings, translated to multiple languages."""


class v:
    """Constants for the Specs generator."""

    APP_NAME = 'gws'
    EXT_PREFIX = APP_NAME + '.ext'
    EXT_DECL_PREFIX = EXT_PREFIX + '.new.'
    EXT_CONFIG_PREFIX = EXT_PREFIX + '.config.'
    EXT_PROPS_PREFIX = EXT_PREFIX + '.props.'
    EXT_OBJECT_PREFIX = EXT_PREFIX + '.object.'
    EXT_COMMAND_PREFIX = EXT_PREFIX + '.command.'

    EXT_COMMAND_API_PREFIX = EXT_COMMAND_PREFIX + 'api.'
    EXT_COMMAND_GET_PREFIX = EXT_COMMAND_PREFIX + 'get.'
    EXT_COMMAND_CLI_PREFIX = EXT_COMMAND_PREFIX + 'cli.'

    EXT_OBJECT_CLASS = 'Object'
    EXT_CONFIG_CLASS = 'Config'
    EXT_PROPS_CLASS = 'Props'

    CLIENT_NAME = 'gc'
    VARIANT_TAG = 'type'
    """Tag property name for Variant types."""
    DEFAULT_VARIANT_TAG = 'default'
    """Default variant tag."""

    ATOMS = ['any', 'bool', 'bytes', 'float', 'int', 'str']

    BUILTINS = ATOMS + ['type', 'object', 'Exception', 'dict', 'list', 'set', 'tuple']

    BUILTIN_TYPES = [
        'Any',
        'Callable',
        'ContextManager',
        'Dict',
        'Enum',
        'Iterable',
        'Iterator',
        'List',
        'Literal',
        'Optional',
        'Protocol',
        'Set',
        'Tuple',
        'TypeAlias',
        'Union',
        # imported in TYPE_CHECKING
        'datetime.datetime',
        'osgeo',
        'sqlalchemy',
        # vendor libs
        'gws.lib.vendor',
        'gws.lib.sa',
    ]

    # those star-imported in gws/__init__.py
    GLOBAL_MODULES = [
        APP_NAME + '.core.const',
        APP_NAME + '.core.util',
    ]

    DEFAULT_EXT_SUPERS = {
        'config': APP_NAME + '.core.types.ConfigWithAccess',
        'props': APP_NAME + '.core.types.Props',
    }

    # prefix for gws.plugin class names
    PLUGIN_PREFIX = APP_NAME + '.plugin'

    # inline comment symbol
    INLINE_COMMENT_SYMBOL = '#:'

    # where we are
    SELF_DIR = os.path.dirname(__file__)

    # path to `/repository-root/app`
    APP_DIR = os.path.abspath(SELF_DIR + '/../..')

    EXCLUDE_PATHS = ['___', '/vendor/', 'test', 'core/ext', '__pycache__']

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
        [CLIENT_NAME, f'/js/src/{CLIENT_NAME}'],
        [f'{APP_NAME}.core', '/gws/core'],
        [f'{APP_NAME}.base', '/gws/base'],
        [f'{APP_NAME}.gis', '/gws/gis'],
        [f'{APP_NAME}.lib', '/gws/lib'],
        [f'{APP_NAME}.server', '/gws/server'],
        [f'{APP_NAME}.helper', '/gws/helper'],
    ]
