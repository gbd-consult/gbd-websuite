from typing import Optional, Literal

import gws


class ModuleConfig(gws.Config):
    enabled: bool = True
    """The module is enabled."""
    threads: int = 0
    """Number of threads for this module."""
    workers: int = 4
    """Number of processes for this module."""


class SpoolConfig(ModuleConfig):
    """Spool server module"""

    jobFrequency: gws.Duration = '3'
    """Background jobs checking frequency."""
    timeout: gws.Duration = '300'
    """Job timeout."""


class WebConfig(ModuleConfig):
    """Web server module"""

    maxRequestLength: int = 10
    """Max request length in megabytes."""
    timeout: gws.Duration = '60'
    """Web server timeout."""


class MapproxyConfig(ModuleConfig):
    """Mapproxy server module"""

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

    mapproxy: Optional[MapproxyConfig]
    """Bundled Mapproxy module."""
    monitor: Optional[MonitorConfig]
    """Monitor configuration."""
    log: Optional[LogConfig]
    """Logging configuration."""
    qgis: QgisConfig
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
    """Server timeout."""
    timeZone: str = 'UTC'
    """Timezone for this server."""


class ConfigTemplateArgs(gws.Data):
    """Arguments for configuration templates."""

    root: gws.Root
    """Root object."""
    inContainer: bool
    """True if we're running in a container."""
    userName: str
    """User name."""
    groupName: str
    """Group name."""
    gwsEnv: dict
    """A dict of GWS environment variables."""
    mapproxyPid: str
    """Mapproxy pid path."""
    mapproxySocket: str
    """Mapproxy socket path."""
    nginxPid: str
    """nginx pid path."""
    spoolPid: str
    """Spooler pid path."""
    spoolSocket: str
    """Spooler socket path."""
    webPid: str
    """Web server pid path."""
    webSocket: str
    """Web server socket path."""
