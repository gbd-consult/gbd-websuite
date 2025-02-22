"""Configuration for embedded servers."""

from typing import Optional, Literal

import gws


class SpoolConfig(gws.Config):
    """Spool server module"""

    disabled: bool = False
    """The module is disabled. (added in 8.2)"""
    enabled: Optional[bool]
    """The module is enabled. (deprecated in 8.2)"""
    workers: int = 4
    """Number of processes for this module."""
    jobFrequency: gws.Duration = '3'
    """Background jobs checking frequency."""
    timeout: gws.Duration = '300'
    """Job timeout. (added in 8.1)"""


class WebConfig(gws.Config):
    """Web server module"""

    disabled: bool = False
    """The module is disabled. (added in 8.2)"""
    enabled: Optional[bool]
    """The module is enabled. (deprecated in 8.2)"""
    workers: int = 4
    """Number of processes for this module."""
    maxRequestLength: int = 10
    """Max request length in megabytes."""
    timeout: gws.Duration = '60'
    """Web server timeout. (added in 8.1)"""


class MapproxyConfig(gws.Config):
    """Mapproxy server module"""

    disabled: bool = False
    """The module is disabled. (added in 8.2)"""
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
    disabled: bool = False
    """The module is disabled. (added in 8.2)"""
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
    level: Literal['ERROR', 'INFO', 'DEBUG'] = 'INFO'
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

    templates: Optional[list[gws.ext.config.template]]
    """Configuration templates."""

    autoRun: str = ''
    """Shell command to run before server start."""
    timeout: gws.Duration = '60'
    """Server timeout. (deprecated in 8.1)"""
    timeZone: str = 'Europe/Berlin'
    """Timezone for this server."""


def is_disabled(cfg):
    # old keys first
    if gws.u.get(cfg, 'enabled') is True:
        return False
    if gws.u.get(cfg, 'enabled') is False:
        return True
    # new keys
    return gws.u.get(cfg, 'disabled')
