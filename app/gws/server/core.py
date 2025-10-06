"""Configuration for embedded servers."""

from typing import Optional, Literal

import gws


class SpoolConfig(gws.Config):
    """Spool server module"""

    enabled: Optional[bool]
    """The module is enabled. (deprecated in 8.2)"""
    workers: int = 4
    """Number of processes for this module."""
    jobFrequency: gws.Duration = '3'
    """Background jobs checking frequency."""
    timeout: gws.Duration = '300'
    """Job timeout."""


class WebConfig(gws.Config):
    """Web server module"""

    enabled: Optional[bool]
    """The module is enabled. (deprecated in 8.2)"""
    workers: int = 4
    """Number of processes for this module."""
    maxRequestLength: int = 10
    """Max request length in megabytes."""
    timeout: gws.Duration = '60'
    """Web server timeout."""


class MapproxyConfig(gws.Config):
    """Mapproxy server module"""

    enabled: Optional[bool]
    """The module is enabled. (deprecated in 8.2)"""
    workers: int = 4
    """Number of processes for this module."""
    host: str = 'localhost'
    """Host to run the module on."""
    port: int = 5000
    """Port number."""
    forceStart: bool = False
    """Start even if no configuration is defined."""


class MonitorConfig(gws.Config):
    """Monitor module configuration."""

    enabled: Optional[bool]
    """The module is enabled. (deprecated in 8.2)"""
    frequency: gws.Duration = '30'
    """Periodic tasks frequency."""
    disableWatch: bool = False
    """Disable file system watching. (added in 8.2)"""
    ignore: Optional[list[gws.Regex]]
    """Ignore paths that match these regexes. (deprecated in 8.2)"""


class QgisConfig(gws.Config):
    """External QGIS server configuration."""

    host: str = 'qgis'
    """Host where the qgis server runs."""
    port: int = 80
    """Port number."""


class LogConfig(gws.Config):
    """Logging configuration"""

    path: str = ''
    """Log path."""
    level: str = 'INFO'
    """Logging level."""


class Config(gws.Config):
    """Server module configuration"""

    mapproxy: Optional[MapproxyConfig]
    """Bundled Mapproxy module."""
    monitor: Optional[MonitorConfig]
    """Monitor configuration."""
    log: Optional[LogConfig]
    """Logging configuration."""
    qgis: Optional[QgisConfig]
    """Qgis server configuration."""
    spool: Optional[SpoolConfig]
    """Spool server module."""
    web: Optional[WebConfig]
    """Web server module."""

    withWeb: bool = True
    """Enable the web server. (added in 8.2)"""
    withSpool: bool = True
    """Enable the spool server. (added in 8.2)"""
    withMapproxy: bool = True
    """Enable the mapproxy server. (added in 8.2)"""
    withMonitor: bool = True
    """Enable the monitor. (added in 8.2)"""

    templates: Optional[list[gws.ext.config.template]]
    """Configuration templates."""

    autoRun: str = ''
    """Shell command to run before the server start. (deprecated in 8.2)"""
    preConfigure: str = ''
    """Shell or python script to run before configuring the server. (added in 8.2)"""
    postConfigure: str = ''
    """Shell or python script to run run after the service has been configured. (added in 8.2)"""
    timeZone: str = 'Europe/Berlin'
    """Timezone for this server."""
