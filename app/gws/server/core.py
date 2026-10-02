"""Configuration for embedded servers."""

from typing import Optional, Literal

import gws


class SpoolConfig(gws.Config):
    """Spool server, which runs background jobs."""

    enabled: Optional[bool]
    """The module is enabled. (deprecated in 8.2)"""
    workers: int = 4
    """Number of spool worker processes."""
    jobFrequency: gws.Duration = '3'
    """Interval at which the spool server checks for new jobs."""
    timeout: gws.Duration = '300'
    """Max. run time of a background job before it is aborted."""


class WebConfig(gws.Config):
    """Web server, which handles client and API requests."""

    enabled: Optional[bool]
    """The module is enabled. (deprecated in 8.2)"""
    workers: int = 4
    """Number of web worker processes."""
    maxRequestLength: int = 10
    """Max. request body size in megabytes."""
    timeout: gws.Duration = '60'
    """Max. time to process a request before it is aborted."""


class MonitorConfig(gws.Config):
    """Monitor, which watches files and runs periodic tasks."""

    enabled: Optional[bool]
    """The module is enabled. (deprecated in 8.2)"""
    frequency: gws.Duration = '30'
    """Default interval of periodic tasks."""
    disableWatch: bool = False
    """Do not reconfigure the server when watched files change."""
    ignore: Optional[list[gws.Regex]]
    """Ignore paths that match these regexes. (deprecated in 8.2)"""


class QgisConfig(gws.Config):
    """External QGIS server configuration."""

    host: str = 'qgis'
    """Host where the QGIS server runs."""
    port: int = 80
    """Port of the QGIS server."""


class LogConfig(gws.Config):
    """Logging configuration."""

    path: str = ''
    """Log path."""
    level: str = 'INFO'
    """Log level."""


class Config(gws.Config):
    """Server processes and logging."""

    mapproxy: Optional[dict]
    """Bundled MapProxy module, ignored. (deprecated in 8.5)"""
    monitor: Optional[MonitorConfig]
    """Monitor configuration."""
    log: Optional[LogConfig]
    """Logging configuration."""
    qgis: Optional[QgisConfig]
    """QGIS server configuration."""
    spool: Optional[SpoolConfig]
    """Spool server module configuration."""
    web: Optional[WebConfig]
    """Web server module configuration."""

    withWeb: bool = True
    """Run the web server."""
    withSpool: bool = True
    """Run the spool server for background jobs."""
    withMapproxy: Optional[bool]
    """Run the MapProxy server, ignored. (deprecated in 8.5)"""
    withMonitor: bool = True
    """Run the monitor."""

    templates: Optional[list[gws.ext.config.template]]
    """Templates for the nginx, uWSGI and syslog configs and the start script."""

    autoRun: str = ''
    """Shell command to run before the server start. (deprecated in 8.2)"""
    preConfigure: str = ''
    """Shell or Python script to run before configuring the server."""
    postConfigure: str = ''
    """Shell or Python script to run after the server has been configured."""
    timeZone: str = 'Europe/Berlin'
    """Time zone of the server."""
