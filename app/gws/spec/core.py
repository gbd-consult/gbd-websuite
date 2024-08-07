class Error(Exception):
    pass


class GeneratorError(Error):
    pass


class ReadError(Error):
    pass


class LoadError(Error):
    pass


class C:
    ATOM = 'ATOM'
    CLASS = 'CLASS'
    CALLABLE = 'CALLABLE'
    CONSTANT = 'CONSTANT'
    DICT = 'DICT'
    ENUM = 'ENUM'
    EXPR = 'EXPR'
    FUNCTION = 'FUNCTION'
    LIST = 'LIST'
    LITERAL = 'LITERAL'
    METHOD = 'METHOD'
    MODULE = 'MODULE'
    OPTIONAL = 'OPTIONAL'
    PROPERTY = 'PROPERTY'
    SET = 'SET'
    TUPLE = 'TUPLE'
    TYPE = 'TYPE'
    UNION = 'UNION'
    UNDEFINED = 'UNDEFINED'
    VARIANT = 'VARIANT'

    EXT = 'EXT'
    COMMAND = 'COMMAND'
    OBJECT = 'OBJECT'
    CONFIG = 'CONFIG'
    PROPS = 'PROPS'


# prefix for our decorators and ext classes
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

TypeId = str

# tag property name for Variant types
VARIANT_TAG = 'type'

# default variant tag
DEFAULT_VARIANT_TAG = 'default'


class Type:
    c: C
    uid: TypeId
    extName: str = ''

    doc: str = ''
    ident: str = ''
    name: str = ''
    pos = ''

    modName: str = ''
    modPath: str = ''

    tArg: TypeId = ''
    tItem: TypeId = ''
    tKey: TypeId = ''
    tModule: TypeId = ''
    tOwner: TypeId = ''
    tReturn: TypeId = ''
    tTarget: TypeId = ''
    tValue: TypeId = ''

    tArgs: list[TypeId] = []
    tItems: list[TypeId] = []
    tSupers: list[TypeId] = []

    tMembers: dict[str, TypeId] = {}
    tProperties: dict[str, TypeId] = {}

    defaultValue = None
    hasDefault: bool = False

    enumText: str = ''
    enumDocs: dict = {}
    enumValues: dict = {}

    literalValues: list = []

    vars: dict[str, TypeId] = {}

    def __init__(self, **kwargs):
        vars(self).update(kwargs)
