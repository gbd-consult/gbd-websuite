"""XML-related utilities and helpers."""

from .parser import from_path, from_string
from .tag import tag
from .error import Error, ParseError, WriteError, BuildError
from . import namespace, util