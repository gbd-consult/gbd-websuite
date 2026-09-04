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


class MonitorConfig(gws.Config):
    """Monitor module configuration."""

    enabled: Optional[bool]
    """The module is enabled. (deprecated in 8.2)"""
    frequency: gws.Duration = '30'
    """Periodic tasks frequency."""
    disableWatch: bool = False
    """Disable file system watching."""
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

    mapproxy: Optional[dict]
    """Bundled Mapproxy module. (deprecated in 8.5)"""
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
    """Enable the web server."""
    withSpool: bool = True
    """Enable the spool server."""
    withMapproxy: Optional[bool]
    """Enable the mapproxy server. (deprecated in 8.5)"""
    withMonitor: bool = True
    """Enable the monitor."""

    templates: Optional[list[gws.ext.config.template]]
    """Configuration templates."""

    autoRun: str = ''
    """Shell command to run before the server start. (deprecated in 8.2)"""
    preConfigure: str = ''
    """Shell or python script to run before configuring the server."""
    postConfigure: str = ''
    """Shell or python script to run run after the service has been configured."""
    timeZone: str = 'Europe/Berlin'
    """Timezone for this server."""
