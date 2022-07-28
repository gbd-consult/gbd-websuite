from typing import Any, List, Dict


class C:
    ALIAS = 'ALIAS'
    ATOM = 'ATOM'
    CLASS = 'CLASS'
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

    COMMAND = 'COMMAND'
    OBJECT = 'OBJECT'
    CONFIG = 'CONFIG'
    PROPS = 'PROPS'


class Meta:
    manifest = None
    manifestPath = ''
    version = ''


TypeId = str


class Type:
    c: C
    uid: TypeId

    doc: str = ''
    ident: str = ''
    extName: str = ''
    name: str = ''
    pos = ''

    tItem: TypeId = ''
    tKey: TypeId = ''
    tModule: TypeId = ''
    tOwner: TypeId = ''
    tReturn: TypeId = ''
    tTarget: TypeId = ''
    tValue: TypeId = ''

    tArgs: List[TypeId] = []
    tItems: List[TypeId] = []
    tSupers: List[TypeId] = []

    tMembers: Dict[str, TypeId] = {}
    tProperties: Dict[str, TypeId] = {}

    default: Any = None
    hasDefault: bool = False

    enumText: str = ''
    enumDocs: dict = {}
    enumValues: dict = {}

    values: List[Any] = []

    vars: Dict[str, TypeId] = {}

    def __init__(self, **kwargs):
        vars(self).update(kwargs)
