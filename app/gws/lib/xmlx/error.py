"""XML-related exceptions."""

import gws


class Error(gws.Error):
    pass


class ParseError(Error):
    pass


class WriteError(Error):
    pass


class NamespaceError(Error):
    pass


class BuildError(Error):
    pass
