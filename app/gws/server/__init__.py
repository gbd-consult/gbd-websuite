"""Configuration and management of embedded servers.

GWS runs several servers in the container: WSGI backend servers (Web and Mapproxy), the Spool server for background jobs and the frontend NGINX proxy.

This module provides configuration and control utilities for these facilities. It handles GWS startups and reloads.

The GWS startup sequence is the following:

- the main script ``bin/gws`` invokes the ``start`` command in :obj:`gws.server.cli`
- CLI delegates to :obj:`gws.server.control`
- control invokes the ``Application`` configuration in :obj:`gws.base.application.core.Object`
- the Application creates a ``ServerManager`` (:obj:`gws.server.manager.Object`)
- the Manager creates configuration files for embedded servers and a startup shell script to start them
- the control is returned to ``bin/gws``, which invokes the startup script
- the script starts the backends and finally NGINX, which keeps running in foreground
"""

from .core import Config
