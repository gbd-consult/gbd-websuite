import os
from typing import Any, Dict, List, Union

ATOMS = ['any', 'bool', 'bytes', 'float', 'int', 'str']

BUILTINS = ATOMS + ['type', 'object', 'Exception', 'Protocol']

# those star-imported in gws/__init__.py
GLOBAL_MODULES = ['gws.core.const', 'gws.core.data', 'gws.core.types', 'gws.core.util', 'gws.core']

# prefix for our decorators and ext classes
GWS_EXT_PREFIX = 'gws.ext'

# prefix for plugin class names
GWS_PLUGIN_PREFIX = 'gws.plugin'

# tag property name for Variant types
GWS_TAG_PROPERTY = 'type'

# path to `/repository-root/app`
APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../../..')

# type for possible ("unchecked") enum values
UNCHECKED_ENUM = 'UNCHECKED_ENUM'

# gws. modules are available in the container,
# but might not be available on the host

try:
    from gws.core.data import Data
except ImportError:
    class Data:  # type: ignore
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

        def __repr__(self):
            return repr(vars(self))

        def get(self, k, default=None):
            return vars(self).get(k, default)

        def __getattr__(self, item):
            return None

try:
    from gws.core import log
except ImportError:
    class _Logger:
        level = 'INFO'

        def set_level(self, level):
            self.level = level

        def log(self, level, *args):
            levels = 'ERROR', 'WARNING', 'INFO', 'DEBUG'
            if levels.index(level) <= levels.index(self.level):
                print(f'SPECGEN: {level}:', *args)

        def error(self, *args): self.log('ERROR', *args)

        def warn(self, *args): self.log('WARNING', *args)

        def info(self, *args): self.log('INFO', *args)

        def debug(self, *args): self.log('DEBUG', *args)


    log = _Logger()  # type: ignore

try:
    from gws.core.const import VERSION
except ImportError:
    with open(APP_DIR + '/../VERSION', 'rt') as fp:
        VERSION = fp.read()  # type: ignore

LiteralValue = Union[str, int, float]

TypeName = str


class Type:
    name = ''


class TAtom(Type):
    def __init__(self, name: str):
        self.name = name


class TDict(Type):
    def __init__(self, key_t: TypeName, value_t: TypeName):
        self.key_t = key_t
        self.value_t = value_t
        self.name = f'Dict[{self.key_t},{self.value_t}]'


class TUnresolvedReference(Type):
    def __init__(self, name: str):
        self.name = name


class TList(Type):
    def __init__(self, item_t: TypeName):
        self.item_t = item_t
        self.name = f'List[{self.item_t}]'


class TLiteral(Type):
    def __init__(self, values: List[LiteralValue]):
        self.values = values
        self.name = '|'.join(repr(v) for v in self.values)


class TOptional(Type):
    def __init__(self, target_t: TypeName):
        self.target_t = target_t
        self.name = f'Optional[{self.target_t}]'


class TTuple(Type):
    def __init__(self, items: List[TypeName]):
        self.items = items
        self.name = '[' + ','.join(self.items) + ']'


class TUnion(Type):
    def __init__(self, items: List[TypeName]):
        self.items = items
        self.name = 'Union[' + ','.join(self.items) + ']'


class TVariant(Type):
    def __init__(self, members: Dict[str, TypeName]):
        self.members = members
        self.name = 'Union[' + ','.join(self.members.values()) + ']'


class TVariantStub(Type):
    def __init__(self, items: List[TypeName], pos: dict):
        self.items = items
        self.pos = pos
        self.name = 'VariantStub[' + ','.join(self.items) + ']'


class TNamedType(Type):
    def __init__(
            self,
            doc: str,
            ident: str,
            name: str,
            pos: dict,
    ):
        self.name = name
        self.ident = ident
        self.pos = pos
        self.doc = doc


class TAlias(TNamedType):
    def __init__(
            self,
            doc: str,
            ident: str,
            pos: dict,
            name: str,
            target_t: TypeName,
    ):
        super().__init__(doc, ident, name, pos)
        self.target_t = target_t


class TCommand(TNamedType):
    def __init__(
            self,
            doc: str,
            ident: str,
            name: str,
            pos: dict,
            arg_t: TypeName,
            cmd_action: str,
            cmd_command: str,
            cmd_method: str,
            cmd_name: str,
            ext_type: str,
            owner_t: TypeName,
            ret_t: TypeName,
    ):
        super().__init__(doc, ident, name, pos)
        self.owner_t = owner_t
        self.cmd_action = cmd_action
        self.cmd_command = cmd_command
        self.cmd_method = cmd_method
        self.cmd_name = cmd_name
        self.ext_type = ext_type
        self.arg_t = arg_t
        self.ret_t = ret_t


class TEnum(TNamedType):
    def __init__(
            self,
            doc: str,
            ident: str,
            name: str,
            pos: dict,
            docs: dict,
            values: dict,
    ):
        super().__init__(doc, ident, name, pos)
        self.docs = docs
        self.values = values


class TObject(TNamedType):
    def __init__(
            self,
            doc: str,
            ident: str,
            name: str,
            pos: dict,
            ext_category: str,
            ext_kind: str,
            ext_type: str,
            super_t: TypeName,
    ):
        super().__init__(doc, ident, name, pos)
        self.super_t = super_t
        self.ext_category = ext_category
        self.ext_kind = ext_kind
        self.ext_type = ext_type
        self.props: dict = {}


class TProperty(TNamedType):
    def __init__(
            self,
            doc: str,
            ident: str,
            name: str,
            pos: dict,
            default: Any,
            has_default: bool,
            owner_t: TypeName,
            property_t: TypeName,
    ):
        super().__init__(doc, ident, name, pos)
        self.has_default = has_default
        self.default = default
        self.owner_t = owner_t
        self.property_t = property_t


class ParserState:
    def __init__(self):
        self.aliases = {}
        self.types = {}


class Error(Exception):
    pass
