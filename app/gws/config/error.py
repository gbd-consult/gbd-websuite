import gws
import gws.types as t


class Error(gws.Error):
    pass


class ParseError(Error):
    pass


class ConfigurationError(Error):
    pass


class LoadError(Error):
    pass


class MapproxyConfigurationError(Error):
    pass
