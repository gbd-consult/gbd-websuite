"""Configuration and control of the embedded servers.

GWS runs several servers in the container: the uWSGI web server, which handles
client and API requests, the uWSGI spool server, which runs background jobs and
the monitor, the frontend NGINX proxy and, in a container, an ``rsyslogd``
daemon for logging. This package configures these servers, creates their
configuration files and the start script, and handles server starts, reloads
and reconfigurations.

Submodules:

- ``core``: the ``server`` section of the application configuration
  (``Config`` and the configs for the web and spool servers, the monitor,
  logging and QGIS).
- ``manager``: the server manager (``gws.ServerManager``), which applies config
  defaults and environment overrides and renders the configuration files and
  the start script from templates.
- ``control``: functions that start, reconfigure and reload the servers, and
  test the configuration.
- ``cli``: the ``server`` command-line commands, which delegate to ``control``.
- ``monitor``: the server monitor (``gws.ServerMonitor``), which watches
  configuration files and runs periodic tasks.
- ``spool``: the spool server application and the job queue functions.
- ``uwsgi_module``: access to the ``uwsgi`` module, which only exists inside
  a uWSGI process.
- ``templates``: default templates for the nginx, uWSGI and syslog configs and
  the start script.

The configuration files are rendered from templates, which can be replaced in
the ``server.templates`` config. These template subjects are used:

- ``server.rsyslog_config``: the embedded ``rsyslogd`` daemon, only in a container
- ``server.uwsgi_config``: the uWSGI backends, the ``uwsgi`` argument holds the
  backend name (``web`` or ``spool``)
- ``server.nginx_config``: the frontend NGINX proxy
- ``server.start_script``: the shell script that starts the servers

Each template receives a :obj:`gws.server.manager.TemplateArgs` object as
arguments. By default, text-only templates from the ``templates`` directory
are used.

The startup sequence is the following:

- the main script ``bin/gws`` invokes the ``server start`` command in :obj:`gws.server.cli`
- the CLI delegates to :obj:`gws.server.control`
- ``control`` configures the application (:obj:`gws.base.application.core.Object`)
  and stores the configuration
- the application creates the server manager (:obj:`gws.server.manager.Object`)
  and the monitor (:obj:`gws.server.monitor.Object`)
- ``control`` makes the manager write the configuration files for the servers
  and the start script
- control returns to ``bin/gws``, which executes the start script
- the script starts ``rsyslogd``, the uWSGI backends and finally NGINX, which
  keeps running in the foreground

On configure, the manager fills in the defaults of the ``server`` config and
its subsections, maps the deprecated ``enabled`` keys to ``withWeb``,
``withSpool`` and ``withMonitor``, and applies overrides from the environment
variables ``GWS_LOG_LEVEL``, ``GWS_WEB_WORKERS`` and ``GWS_SPOOL_WORKERS``.

Besides the start, ``control`` supports these workflows:

- reconfigure (``server reconfigure``): configure the
  application, store the configuration, rewrite the server configs, empty the
  transient directory, reload the uWSGI backends and NGINX
- reload (``server reload``): empty the transient directory,
  reload the uWSGI backends and NGINX
- configure (``server configure``, for debugging): configure the application
  and store the configuration
- config test (``server configtest``, for debugging): configure or only parse
  the configuration and log the report

The spool server loads the stored configuration and starts the monitor, if
``withMonitor`` is set. A uWSGI timer calls the monitor every few seconds. When watched files change, or when an object calls
``schedule_reload``, the monitor configures and stores the configuration (on
a reconfigure) and reloads the web and spool backends. It does not rewrite the
server configs and does not reload NGINX.

Example::

    server {
        withSpool true
        spool { workers 2 timeout 600 }
        web { workers 8 maxRequestLength 50 }
        log { level "DEBUG" }
        qgis { host "qgis" port 80 }
        timeZone "Europe/Berlin"
    }

Example::

    root.app.monitor.register_periodic_task(self, frequency=60)
    root.app.monitor.schedule_reload(with_reconfigure=True)
"""

from .core import Config
