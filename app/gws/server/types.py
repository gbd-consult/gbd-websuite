import gws
import gws.types as t


class SrvConfig(t.Config):
    enabled: bool = True  #: the module is enabled
    threads: int = 0  #: number of threads for this module
    workers: int = 4  #: number of processes for this module


class SpoolConfig(SrvConfig):
    """Spool server module"""

    jobFrequency: t.duration = 3  #: background jobs checking frequency
    monitorFrequency: t.duration = 30  #: filesystem changes check frequency


class WebConfig(SrvConfig):
    """Web server module"""
    maxRequestLength: int = 10  #: max request length in megabytes
    pass


class MapproxyConfig(SrvConfig):
    """Mapproxy server module"""

    host: str = 'localhost'
    port: int = 5000


class QgisConfig(SrvConfig):
    """Bundled QGIS server module"""

    host: str = 'localhost'
    port: int = 4000
    maxRequests: int = 6  #: max concurrent requests to this server

    debug: int = 0  #: QGIS_DEBUG (env. variable)
    serverLogLevel: int = 2  #: QGIS_SERVER_LOG_LEVEL (env. variable)
    serverCacheSize: int = 10000000  #: QGIS_SERVER_CACHE_SIZE (env. variable)
    maxCacheLayers: int = 4000  #: MAX_CACHE_LAYERS (env. variable)
    searchPathsForSVG: t.Optional[t.List[t.dirpath]]  #: searchPathsForSVG (ini setting)
    legend: t.Optional[dict]  #: default legend settings


class Config(t.Config):
    """Server module configuation"""

    autoRun: str = ''  #: shell command to run before server start
    logLevel: gws.log.Level = 'INFO'  #: logging level
    mapproxy: MapproxyConfig = {}  #: bundled Mapproxy module
    qgis: QgisConfig = {}  #: bundled Qgis module
    spool: SpoolConfig = {}  #: spool server module
    timeout: t.duration = 60  #: server timeout
    web: WebConfig = {}  #: web server module
