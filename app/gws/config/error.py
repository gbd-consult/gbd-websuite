import gws


class Error(gws.Error):
    pass


class ParseError(Error):
    pass


class LoadError(Error):
    pass


class MapproxyConfigError(Error):
    pass
