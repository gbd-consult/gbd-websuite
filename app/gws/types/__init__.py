"""Base types.

This module reimports most commonly used ``typing`` types and provides
a custom ``Enum`` implementation.

It should imported as ``t`` elsewhere ::

    import gws.types as t

    def some_func(arg: t.Optional[int]) -> t.Iterable[str]...
"""

import enum

from typing import (
    cast,
    Any,
    ContextManager,
    Literal,
    Optional,
    Union,
)

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)


class Enum(enum.Enum):
    """Enumeration type.

    Despite being declared as extending ``Enum`` (for IDE support), this class is actually just a simple object
    and intended to be used as a collection of attributes. It doesn't provide any ``Enum``-specific utilities.

    The rationale behind this is that we need ``Enum`` members (e.g. ``Color.RED``) to be scalars,
    and not complex objects as in the standard ``Enum``.
    """
    pass


# hack to make Enum a simple object
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
    'Sequence',
    'Union',
    'Enum',
]
