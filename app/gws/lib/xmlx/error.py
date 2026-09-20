"""XML-related exceptions."""

import gws


class Error(gws.Error):
    """Base class for XML errors."""


class ParseError(Error):
    """Malformed input, forbidden constructs (entity declarations) or undecodable bytes."""


class WriteError(Error):
    """Invalid or prefixed element or attribute name in a tree being serialized."""


class NamespaceError(Error):
    """Unknown or conflicting namespace."""


class BuildError(Error):
    """Invalid argument to ``tag()``."""
