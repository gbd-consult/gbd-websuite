"""Configuration for embedded servers."""

from typing import Optional, Literal

import gws


class SpoolConfig(gws.Config):
    """Spool server module"""

    enabled: bool = True
    """The module is enabled."""
    threads: int = 0
    """Number of threads for this module."""
    workers: int = 4
    """Number of processes for this module."""
    jobFrequency: gws.Duration = '3'
    """Background jobs checking frequency."""
    timeout: gws.Duration = '300'
    """Job timeout."""


class WebConfig(gws.Config):
    """Web server module"""

    enabled: bool = True
    """The module is enabled."""
    threads: int = 0
    """Number of threads for this module."""
    workers: int = 4
    """Number of processes for this module."""
    maxRequestLength: int = 10
    """Max request length in megabytes."""
    timeout: gws.Duration = '60'
    """Web server timeout."""


class MapproxyConfig(gws.Config):
    """Mapproxy server module"""

    enabled: bool = True
    """The module is enabled."""
    threads: int = 0
    """Number of threads for this module."""
    workers: int = 4
    """Number of processes for this module."""
    host: str = 'localhost'
    """Host to run the module on."""
    port: int = 5000
    """Port number."""
    forceStart: bool = False
    """Start even if no configuration is defined."""


class MonitorConfig(gws.Config):
    enabled: bool = True
    """The module is enabled."""
    frequency: gws.Duration = '30'
    """Filesystem changes check frequency."""
    ignore: Optional[list[gws.Regex]]
    """Ignore paths that match these regexes."""


class QgisConfig(gws.Config):
    """QGIS server config"""

    host: str = 'qgis'
    """Host where the qgis server runs."""
    port: int = 80
    """Port number."""


class LogConfig(gws.Config):
    """Logging configuration"""

    path: str = ''
    """Log path."""
    level: Literal['ERROR', 'INFO', 'DEBUG'] = 'INFO'
    """Logging level."""


class Config(gws.Config):
    """Server module configuration"""

    mapproxy: Optional[MapproxyConfig] = {}
    """Bundled Mapproxy module."""
    monitor: Optional[MonitorConfig] = {}
    """Monitor configuration."""
    log: Optional[LogConfig] = {}
    """Logging configuration."""
    qgis: Optional[QgisConfig] = {}
    """Qgis server configuration."""
    spool: Optional[SpoolConfig] = {}
    """Spool server module."""
    web: Optional[WebConfig] = {}
    """Web server module."""

    templates: Optional[list[gws.ext.config.template]]
    """Configuration templates."""

    autoRun: str = ''
    """Shell command to run before server start."""
    timeout: gws.Duration = '60'
    """Server timeout."""
    timeZone: str = 'UTC'
    """Timezone for this server."""
