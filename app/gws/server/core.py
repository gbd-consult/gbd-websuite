from typing import Optional, Literal

import gws


class ModuleConfig(gws.Config):
    enabled: bool = True
    """the module is enabled"""
    threads: int = 0
    """number of threads for this module"""
    workers: int = 4
    """number of processes for this module"""


class SpoolConfig(ModuleConfig):
    """Spool server module"""

    jobFrequency: gws.Duration = '3'
    """background jobs checking frequency"""


class WebConfig(ModuleConfig):
    """Web server module"""

    maxRequestLength: int = 10
    """max request length in megabytes"""


class MapproxyConfig(ModuleConfig):
    """Mapproxy server module"""

    host: str = 'localhost'
    """host to run the module on"""
    port: int = 5000
    """port number"""
    forceStart: bool = False
    """start even if no configuration is defined"""


class MonitorConfig(gws.Config):
    enabled: bool = True
    """the module is enabled"""
    frequency: gws.Duration = '30'
    """filesystem changes check frequency"""
    ignore: Optional[list[gws.Regex]]
    """ignore paths that match these regexes"""


class QgisConfig(gws.Config):
    """QGIS server config"""

    host: str = 'qgis'
    """host where the qgis server runs"""
    port: int = 80
    """port number"""


class LogConfig(gws.Config):
    """Logging configuration"""

    path: str = ''
    """log path"""
    level: Literal['ERROR', 'INFO', 'DEBUG'] = 'INFO'
    """logging level"""


class Config(gws.Config):
    """Server module configuration"""

    mapproxy: MapproxyConfig = {}  # type: ignore
    """bundled Mapproxy module"""
    monitor: MonitorConfig = {}  # type: ignore
    """monitor configuration"""
    log: LogConfig = {}  # type: ignore
    """logging configuration"""
    qgis: QgisConfig = {}  # type: ignore
    """bundled Qgis module"""
    spool: SpoolConfig = {}  # type: ignore
    """spool server module"""
    web: WebConfig = {}  # type: ignore
    """web server module"""

    autoRun: str = ''
    """shell command to run before server start"""
    timeout: gws.Duration = '60'
    """server timeout"""
    timeZone: str = 'UTC'
    """timezone for this server"""
