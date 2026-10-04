"""Core data structures and constants for the specs."""

from typing import TypeAlias, Any
import os


class Error(Exception):
    """Base class for spec errors."""

    pass


class GeneratorError(Error):
    """Raised when the spec generator fails."""

    pass


class ReadError(Error):
    """Raised when a value does not match its spec type.

    The arguments are the message and the offending value. With verbose errors,
    a ``gws.ConfigErrorInfo`` object is added as the third argument.
    """

    pass


class LoadError(Error):
    """Raised when a class cannot be loaded from its module."""

    pass


class c:
    """Type kinds, the values of ``Type.c``."""

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
"""Type kind, one of the constants in ``c``."""
TypeUid: TypeAlias = str
"""Type unique identifier, a string that identifies the type."""


class Type:
    """Spec type record, describes a single type, property, method or module.

    Which fields are populated depends on the type kind ``c``.
    """

    c: TypeKind
    """Type kind, one of the constants in ``c``."""
    uid: TypeUid
    """Type unique identifier, a string that identifies the type."""

    extName: str = ''
    """``gws.ext`` name of an extension type, variant or command method, if any."""

    title: str = ''
    """Documentation title, filled from the strings by ``get_config_types``."""
    doc: str = ''
    """Documentation string for the type."""
    ident: str = ''
    """Source code identifier (class, property or method name), used in the documentation."""
    name: str = ''
    """Qualified name of a named type, e.g. ``gws.base.layer.core.Config``."""
    pos: str = ''
    """Source code position of the definition, as ``path:line``."""

    modName: str = ''
    """Name of the module that defines this type."""
    modPath: str = ''
    """Path to the module that defines this type."""

    tArg: TypeUid = ''
    """For ``METHOD`` types, type uid of the last (request) argument."""
    tItem: TypeUid = ''
    """For ``LIST`` and ``SET`` types, type uid of the item."""
    tKey: TypeUid = ''
    """For ``DICT`` types, type uid of the key."""
    tModule: TypeUid = ''
    """Type uid of the type's module."""
    tOwner: TypeUid = ''
    """For ``PROPERTY`` and ``METHOD`` types, type uid of the owning class."""
    tReturn: TypeUid = ''
    """For ``METHOD`` types, type uid of the return value."""
    tTarget: TypeUid = ''
    """For ``TYPE``, ``EXT`` and ``OPTIONAL`` types, type uid of the target type."""
    tValue: TypeUid = ''
    """For ``PROPERTY`` types, type uid of the value; for ``DICT`` types, type uid of the dict values."""

    tArgs: list[TypeUid] = []
    """For ``METHOD`` types, type uids of the arguments."""
    tItems: list[TypeUid] = []
    """For ``UNION``, ``TUPLE`` and ``CALLABLE`` types, type uids of the items."""
    tSupers: list[TypeUid] = []
    """For ``CLASS`` types, type uids of the base classes."""
    tMembers: dict[str, TypeUid] = {}
    """For ``VARIANT`` types, member type uids keyed by the ``type`` tag."""
    tProperties: dict[str, TypeUid] = {}
    """For ``CLASS`` types, property type uids keyed by property name, including inherited properties."""

    defaultValue: Any = None
    """Default value for a property."""
    defaultExpression: Any = None
    """Unevaluated default of a property (a constant or enum reference), evaluated by the normalizer."""
    hasDefault: bool = False
    """True if the type has a default value."""
    constValue: Any = None
    """Constant value for a constant type."""

    enumDocs: dict = {}
    """For ``ENUM`` types, member docstrings keyed by member name."""
    enumValues: dict = {}
    """For ``ENUM`` types, member values keyed by member name."""

    literalValues: list = []
    """For ``LITERAL`` types, the allowed values."""

    isConfig: bool = False
    """True if the type is reachable from the application ``Config``."""


def make_type(args: dict):
    """Create a ``Type`` object.

    Args:
        args: Attribute values for the type.

    Returns:
        A new ``Type`` object.
    """

    typ = Type()
    vars(typ).update(args)
    return typ


class Chunk:
    """Source code chunk, the core packages or a plugin with their source files."""

    name: str
    """Name of the chunk."""
    sourceDir: str
    """Source directory of the chunk."""
    bundleDir: str
    """Directory where the client bundle of the chunk is stored."""
    paths: dict[str, list[str]]
    """Source file paths, grouped by file kind (``python``, ``ts``, ``css``, ``theme``, ``strings``)."""
    exclude: list[str]
    """Path fragments to exclude from the chunk."""


class SpecData:
    """Specs data, produced by the generator and loaded by the runtime."""

    meta: dict
    """Build-time metadata: ``version``, ``manifestPath`` and ``manifest``."""
    chunks: list[Chunk]
    """Source code chunks of the application and its plugins."""
    serverTypes: list[Type]
    """Types used by the server: configuration, request and response types, ext objects and command methods."""
    strings: dict[str, dict[str, str]]
    """Documentation strings keyed by language code and type uid."""


class v:
    """Constants for the spec generator and runtime."""

    APP_NAME = 'gws'
    """Application package name."""
    EXT_PREFIX = APP_NAME + '.ext'
    """Prefix of all extension names."""
    EXT_DECL_PREFIX = EXT_PREFIX + '.new.'
    """Prefix of ``gws.ext.new`` declarations in the sources."""
    EXT_CONFIG_PREFIX = EXT_PREFIX + '.config.'
    """Prefix of extension config names."""
    EXT_PROPS_PREFIX = EXT_PREFIX + '.props.'
    """Prefix of extension props names."""
    EXT_OBJECT_PREFIX = EXT_PREFIX + '.object.'
    """Prefix of extension object names."""
    EXT_COMMAND_PREFIX = EXT_PREFIX + '.command.'
    """Prefix of command names."""

    EXT_COMMAND_API_PREFIX = EXT_COMMAND_PREFIX + 'api.'
    """Prefix of API command names."""
    EXT_COMMAND_GET_PREFIX = EXT_COMMAND_PREFIX + 'get.'
    """Prefix of web GET command names."""
    EXT_COMMAND_CLI_PREFIX = EXT_COMMAND_PREFIX + 'cli.'
    """Prefix of CLI command names."""

    EXT_OBJECT_CLASS = 'Object'
    """Default object class name in a ``gws.ext.new`` declaration."""
    EXT_CONFIG_CLASS = 'Config'
    """Default config class name in a ``gws.ext.new`` declaration."""
    EXT_PROPS_CLASS = 'Props'
    """Default props class name in a ``gws.ext.new`` declaration."""

    CLIENT_NAME = 'gc'
    """Name of the client chunk."""
    VARIANT_TAG = 'type'
    """Name of the property that selects the member of a variant."""
    DEFAULT_VARIANT_TAG = 'default'
    """Variant member used when the tag property is missing."""

    ATOMS = ['any', 'bool', 'bytes', 'float', 'int', 'str']
    """Names of atomic types."""

    BUILTINS = ATOMS + ['type', 'object', 'Exception', 'dict', 'list', 'set', 'tuple']
    """Built-in names, registered as ``ATOM`` types."""

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
    """Names from ``typing`` and foreign modules that are treated as built-in."""

    # those star-imported in gws/__init__.py
    GLOBAL_MODULES = [
        APP_NAME + '.core.const',
        APP_NAME + '.core.util',
    ]
    """Modules whose names are available directly as ``gws.<Name>``."""

    DEFAULT_EXT_SUPERS = {
        'config': APP_NAME + '.core.types.ConfigWithAccess',
        'props': APP_NAME + '.core.types.Props',
    }
    """Default base classes for synthesized ext config and props classes."""

    # prefix for gws.plugin class names
    PLUGIN_PREFIX = APP_NAME + '.plugin'

    # inline comment symbol
    INLINE_COMMENT_SYMBOL = '#:'

    # where we are
    SELF_DIR = os.path.dirname(__file__)

    # path to `/repository-root/app`
    APP_DIR = os.path.abspath(SELF_DIR + '/../..')

    EXCLUDE_PATHS = ['___', '/vendor/', 'test', 'core/ext', '__pycache__']
    """Path fragments of source files the generator skips."""

    FILE_KINDS = [
        ['.py', 'python'],
        ['/index.ts', 'ts'],
        ['/index.tsx', 'ts'],
        ['/index.css.js', 'css'],
        ['.theme.css.js', 'theme'],
        ['/strings.ini', 'strings'],
    ]
    """Source file suffixes and the file kinds they map to."""

    PLUGIN_DIR = '/gws/plugin'
    """Directory of the built-in plugins, relative to the app directory."""

    SYSTEM_CHUNKS = [
        [CLIENT_NAME, f'/js/src/{CLIENT_NAME}'],
        [f'{APP_NAME}.core', '/gws/core'],
        [f'{APP_NAME}.base', '/gws/base'],
        [f'{APP_NAME}.gis', '/gws/gis'],
        [f'{APP_NAME}.lib', '/gws/lib'],
        [f'{APP_NAME}.server', '/gws/server'],
        [f'{APP_NAME}.helper', '/gws/helper'],
    ]
    """Names and source directories of the system chunks."""
