import gws
import gws.types as t


class SrvConfig(t.Config):
    #: the module is enabled
    enabled: bool = True
    #: number of threads for this module
    threads: int = 0
    #: number of processes for this module
    workers: int = 4


class SpoolConfig(SrvConfig):
    """Spool server module"""

    #: background jobs checking frequency
    jobFrequency: t.duration = 3
    #: filesystem changes check frequency
    monitorFrequency: t.duration = 30


class WebConfig(SrvConfig):
    """Web server module"""
    pass


class MapproxyConfig(SrvConfig):
    """Mapproxy server module"""

    host: str = 'localhost'
    port: int = 5000


class QgisConfig(SrvConfig):
    """Bundled QGIS server module"""

    host: str = 'localhost'
    port: int = 4000
    #: max concurrent requests to this server
    maxRequests: int = 6

    #: QGIS_DEBUG (env. variable)
    debug: int = 0
    #: QGIS_SERVER_LOG_LEVEL (env. variable)
    serverLogLevel: int = 2
    #: QGIS_SERVER_CACHE_SIZE (env. variable)
    serverCacheSize: int = 10000000
    #: MAX_CACHE_LAYERS (env. variable)
    maxCacheLayers: int = 4000
    #: searchPathsForSVG (ini setting)
    searchPathsForSVG: t.Optional[t.List[t.dirpath]]



class Config(t.Config):
    """Server module configuation"""

    #: bundled Mapproxy module
    mapproxy: MapproxyConfig = {}
    #: bundled Qgis module
    qgis: QgisConfig = {}
    #: spool server module
    spool: SpoolConfig = {}
    #: web server module
    web: WebConfig = {}
    #: logging level
    logLevel: gws.log.Level = 'INFO'
    #: server timeout
    timeout: t.duration = 60
