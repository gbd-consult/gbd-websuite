"""Functions to start, reconfigure and reload the servers."""

import gws
import gws.config
import gws.lib.datetimex
import gws.lib.osx
import gws.lib.watcher

# see bin/gws
_SERVER_START_SCRIPT = f'{gws.c.VAR_DIR}/server.sh'

_PID_PATHS = {
    'web': f'{gws.c.PIDS_DIR}/web.uwsgi.pid',
    'spool': f'{gws.c.PIDS_DIR}/spool.uwsgi.pid',
    'nginx': f'{gws.c.PIDS_DIR}/nginx.pid',
}


def start(manifest_path='', config_path=''):
    """Configure the application and write the server configuration files and the start script.

    Called once on the container start. The start script itself is executed by ``bin/gws``.
    Exits the process with code 1 if the web server is already running.

    Args:
        manifest_path: Path to the application manifest.
        config_path: Path to the configuration file.

    Raises:
        ``gws.ConfigurationError``: If the configuration fails.
    """
    if app_is_running('web'):
        gws.log.error(f'server already running')
        gws.u.exit(1)
    root = configure_and_store(manifest_path, config_path, is_starting=True)
    root.app.serverMgr.create_server_configs(gws.c.SERVER_DIR, _SERVER_START_SCRIPT, _PID_PATHS)


def reconfigure(manifest_path='', config_path=''):
    """Configure the application, rewrite the server configuration files and reload all servers.

    Exits the process with code 1 if the web server is not running.

    Args:
        manifest_path: Path to the application manifest.
        config_path: Path to the configuration file.

    Raises:
        ``gws.ConfigurationError``: If the configuration fails.
    """
    if not app_is_running('web'):
        gws.log.error(f'server not running')
        gws.u.exit(1)
    root = configure_and_store(manifest_path, config_path, is_starting=False)
    root.app.serverMgr.create_server_configs(gws.c.SERVER_DIR, _SERVER_START_SCRIPT, _PID_PATHS)
    reload_all()


def configure_and_store(manifest_path='', config_path='', is_starting=False):
    """Configure the application and store the configuration, so that the servers can load it.

    Args:
        manifest_path: Path to the application manifest.
        config_path: Path to the configuration file.
        is_starting: True on the server start, runs the ``server.autoRun`` command before initialization.

    Returns:
        The configured root object.

    Raises:
        ``gws.ConfigurationError``: If the configuration fails.
    """
    root = configure(manifest_path, config_path, is_starting)
    gws.config.store(root)
    return root


def configure(manifest_path='', config_path='', is_starting=False):
    """Configure the application and log the configuration report.

    If the configuration fails and the manifest enables ``withFallbackConfig``, a minimal fallback configuration is used.

    Args:
        manifest_path: Path to the application manifest.
        config_path: Path to the configuration file.
        is_starting: True on the server start, runs the ``server.autoRun`` command before initialization.

    Returns:
        The configured root object.

    Raises:
        ``gws.ConfigurationError``: If the configuration fails.
    """
    def _pre_init(ld: gws.config.loader.Object):
        autorun = gws.u.get(ld.config, 'server.autoRun')
        if autorun:
            gws.log.info(f'AUTORUN: {autorun!r}')
            gws.lib.osx.run(autorun, echo=True)

    hooks = []
    if is_starting:
        hooks.append(['preInitialize', _pre_init])

    cr = gws.config.configure(
        manifest_path=manifest_path,
        config_path=config_path,
        fallback_config=_FALLBACK_CONFIG,
        hooks=hooks,
    )
    gws.config.log_report(cr)
    if not cr.root:
        raise gws.ConfigurationError('configuration failed')
    return gws.u.require(cr.root)


def config_test(
        manifest_path='',
        config_path='',
        dirs_to_watch='',
        with_parse_only=False,
        with_watch=False
):
    """Configure or parse the configuration and log the report.

    In the watch mode, the test is repeated whenever a file in the watched directories changes,
    and the function never returns.

    Args:
        manifest_path: Path to the application manifest.
        config_path: Path to the configuration file.
        dirs_to_watch: Directories to watch, ``/data`` by default.
        with_parse_only: Only parse the configuration, do not configure the objects.
        with_watch: Keep watching the directories and repeat the test on changes.
    """

    def _check(*args):
        gws.log.info('=' * 80)
        gws.log.info(f'TESTING CONFIGURATION...')
        gws.log.info('=' * 80)
        if with_parse_only:
            cr = gws.config.parse(manifest_path, config_path)
        else:
            cr = gws.config.configure(manifest_path, config_path)
        gws.config.log_report(cr)

    _check()

    if not with_watch:
        return

    w = gws.lib.watcher.new(_check)
    dirs = gws.u.to_list(dirs_to_watch or '/data')
    for d in dirs:
        w.add_directory(d, recursive=True)
    w.start()

    while True:
        gws.u.sleep(3)



##

def reload_all():
    """Empty the transient directory and reload the spool and web servers and NGINX.

    Returns:
        Always True.
    """
    gws.lib.osx.run(['rm', '-fr', gws.c.TRANSIENT_DIR])
    gws.u.ensure_system_dirs()

    reload_app('spool')
    reload_app('web')

    reload_nginx()
    return True


def reload_app(srv):
    """Reload a uWSGI backend, if it is running.

    Args:
        srv: Backend name, ``web`` or ``spool``.
    """
    if not app_is_running(srv):
        gws.log.debug(f'reload: {srv=} not running')
        return
    gws.log.info(f'reloading {srv}...')
    gws.lib.osx.run(['uwsgi', '--reload', _PID_PATHS[srv]])


def reload_nginx():
    """Reload the NGINX configuration."""
    gws.log.info(f'reloading nginx...')
    gws.lib.osx.run(['nginx', '-c', gws.c.SERVER_DIR + '/nginx.conf', '-s', 'reload'])


def app_is_running(srv):
    """Check whether a server is running, by its pid file.

    Args:
        srv: Server name, ``web``, ``spool`` or ``nginx``.

    Returns:
        True if the pid from the pid file belongs to a running process.
    """
    try:
        with open(_PID_PATHS[srv]) as fp:
            pid = int(fp.read())
    except (FileNotFoundError, ValueError):
        pid = 0
    if pid == 0:
        return False
    gws.log.debug(f'found {pid=} for {srv=}')
    return pid in gws.lib.osx.running_pids()


##


_FALLBACK_CONFIG = gws.Config(
    server=gws.Config(
        timeZone="Europe/Berlin",
        monitor=gws.Config(disabled=True),
        log=gws.Config(level='INFO'),
        qgis=gws.Config(host='qgis', port=80),
        spool=gws.Config(disabled=True),
        web=gws.Config(disabled=False, workers=1, timeout=60, maxRequestLength=10),
    )
)
