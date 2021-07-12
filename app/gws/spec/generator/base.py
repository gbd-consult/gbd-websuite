BUILTINS = 'any', 'bool', 'bytes', 'float', 'int', 'str'

# those star-imported in gws/__init__.py
GLOBAL_MODULES = ['gws.core.const.', 'gws.core.data.', 'gws.core.types.', 'gws.core.util.', 'gws.core.']


class ABC:
    alias = 'alias'
    command = 'command'
    descriptor = 'descriptor'
    enum = 'enum'
    module = 'module'
    object = 'object'
    property = 'property'
    stub = 'stub'


class BUILTIN:
    any = 'any'
    bool = 'bool'
    bytes = 'bytes'
    float = 'float'
    int = 'int'
    str = 'str'


class T:
    dict = 'dict'
    list = 'list'
    literal = 'literal'
    optional = 'optional'
    tuple = 'tuple'
    union = 'union'
    variant = 'variant'

    unchecked_variant = 'unchecked_variant'
    unchecked_enum = 'unchecked_enum'


GWS_EXT_PREFIX = 'gws.ext.'

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


class Error(Exception):
    pass


def warn(*args):
    print('WARN', args)
