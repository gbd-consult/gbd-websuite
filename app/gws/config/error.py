import gws
import gws.types as t


class Error(gws.Error):
    pass


class ParseError(Error):
    pass


class ConfigError(Error):
    pass


class LoadError(Error):
    pass


class MapproxyConfigError(Error):
    pass
