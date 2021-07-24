"""Base types"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

# Variant is tagged Union, discriminated by the 'type' property
# mypy doesn't accept aliases to special forms,
# so we don't use Variant = Union
# instead, we type Variants as Unions, but place 'Variant' in the comment string
# so that the spec generator can handle them

# Variant = Union

# @TODO upgrade to py3.8
try:
    from typing import Literal
except ImportError:
    from .vendor.typing_extensions import Literal  # type: ignore

try:
    from typing import Protocol
except ImportError:
    from .vendor.typing_extensions import Protocol  # type: ignore

# We cannot use the standard Enum, because after "class Color(Enum): RED = 1"
# the value of Color.RED is like {'_value_': 1, '_name_': 'RED', '__objclass__': etc}
# and we need it to be 1, literally (that's what we'll get from the client)

# class Enum:
#     pass
#

# avoid 'unused imports'
__all__ = ['Any', 'Callable', 'Dict', 'Enum', 'List', 'Literal', 'Optional', 'Protocol', 'Set', 'Tuple', 'Union', 'cast']
