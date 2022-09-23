import gws


class Error(gws.Error):
    pass


class ParseError(Error):
    pass


class WriteError(Error):
    pass


class BuildError(Error):
    pass
