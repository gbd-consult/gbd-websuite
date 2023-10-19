"""Base types"""

import enum

from typing import (
    cast,
    Any,
    ContextManager,
    Literal,
    Optional,
    Protocol,
    TypeAlias,
    Union,
)

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)



# Variant is tagged Union, discriminated by the 'type' property
# mypy doesn't accept aliases to special forms,
# so we don't use Variant = Union
# instead, we type Variants as Unions, but place 'Variant' in the comment string
# so that the spec generator can handle them

# Variant = Union


# We cannot use the standard Enum, because after "class Color(Enum): RED = 1"
# the value of Color.RED is like {'_value_': 1, '_name_': 'RED', '__objclass__': etc}
# and we need it to be 1, literally (that's what we'll get from the client)

class Enum(enum.Enum):
    pass


globals()['Enum'] = type('Enum', (), {})

__all__ = [
    'cast',
    'Any',
    'Callable',
    'ContextManager',
    'Iterable',
    'Iterator',
    'Literal',
    'Mapping',
    'Optional',
    'Protocol',
    'Sequence',
    'TypeAlias',
    'Union',
    'Enum',
]
