"""Configuration manager for embedded servers.

This object creates configuration files for embedded servers and the server startup script.

The configuration is template-based, there are following template subjects defined:

- ``server.rsyslog_config`` - for the embedded ``rsyslogd`` daemon
- ``server.uwsgi_config`` - for backend uWSGI servers (the ``uwsgi`` argument contains the specific backend name)
- ``server.nginx_config`` - for the frontend NGINX proxy

Each template receives a :obj:`TemplateArgs` object as arguments.

By default, the Manager uses text-only templates from the ``templates`` directory.
"""

from typing import cast

import os
import re

import gws
import gws.base.web
import gws.config
import gws.config.util
import gws.gis.mpx.config
import gws.lib.osx

from . import core


class TemplateArgs(gws.TemplateArgs):
    """Arguments for configuration templates."""

    root: gws.Root
    """Root object."""
    serverDir: str
    """Absolute path to app/server directory."""
    gwsEnv: dict
    """A dict of GWS environment variables."""
    inContainer: bool
    """True if we're running in a container."""
    uwsgi: str
    """uWSGI backend name."""
    userName: str
    """User name."""
    groupName: str
    """User group name."""
    homeDir: str
    """User home directory."""
    mapproxyConfig: str
    """Mapproxy config path."""
    mapproxyPid: str
    """Mapproxy pid path."""
    mapproxySocket: str
    """Mapproxy socket path."""
    nginxConfig: str
    """nginx config path."""
    nginxPid: str
    """nginx pid path."""
    spoolConfig: str
    """Spooler config path."""
    spoolPid: str
    """Spooler pid path."""
    spoolSocket: str
    """Spooler socket path."""
    syslogConfig: str
    """Syslog config path."""
    syslogPid: str
    """Syslog pid path."""
    webConfig: str
    """Web server config path."""
    webPid: str
    """Web server pid path."""
    webSocket: str
    """Web server socket path."""


_DEFAULT_BASE_TIMEOUT = 60
_DEFAULT_SPOOL_TIMEOUT = 300

_SERVER_DIR = gws.u.dirname(__file__)

_DEFAULT_TEMPLATES = [
    gws.Config(type='text', subject='server.rsyslog_config', path=f'{_SERVER_DIR}/templates/rsyslog_config.cx.txt'),
    gws.Config(type='text', subject='server.nginx_config', path=f'{_SERVER_DIR}/templates/nginx_config.cx.txt'),
    gws.Config(type='text', subject='server.uwsgi_config', path=f'{_SERVER_DIR}/templates/uwsgi_config.cx.txt'),
    gws.Config(type='text', subject='server.start_script', path=f'{_SERVER_DIR}/templates/start_script.cx.txt'),
]


class Object(gws.ServerManager):
    def configure(self):
        self.config = self._add_defaults(self.config, 'gws.server.core.Config')
        self.config.log = self._add_defaults(self.config.log, 'gws.server.core.LogConfig')
        self.config.mapproxy = self._add_defaults(self.config.mapproxy, 'gws.server.core.MapproxyConfig')
        self.config.monitor = self._add_defaults(self.config.monitor, 'gws.server.core.MonitorConfig')
        self.config.qgis = self._add_defaults(self.config.qgis, 'gws.server.core.QgisConfig')
        self.config.spool = self._add_defaults(self.config.spool, 'gws.server.core.SpoolConfig')
        self.config.web = self._add_defaults(self.config.web, 'gws.server.core.WebConfig')

        # deprecated 'enabled' keys
        if self.config.mapproxy.enabled is False:
            self.config.withMapproxy = False
        if self.config.spool.enabled is False:
            self.config.withSpool = False
        if self.config.web.enabled is False:
            self.config.withWeb = False
        if self.config.monitor.enabled is False:
            self.config.withMonitor = False

        self.configure_environment()
        self.configure_templates()

    def _add_defaults(self, value, type_name):
        return gws.u.merge(self.root.specs.read({}, type_name), value)

    def configure_environment(self):
        """Overwrite config values from the environment."""

        cfg = cast(core.Config, self.config)
        p = gws.env.GWS_LOG_LEVEL
        if p:
            cfg.log.level = p
        p = gws.env.GWS_WEB_WORKERS
        if p:
            cfg.web.workers = int(p)
        p = gws.env.GWS_SPOOL_WORKERS
        if p:
            cfg.spool.workers = int(p)

    def configure_templates(self):
        gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

    def create_server_configs(self, target_dir, script_path, pid_paths):
        ui = gws.lib.osx.user_info(gws.c.UID, gws.c.GID)

        args = TemplateArgs(
            root=self.root,
            serverDir=_SERVER_DIR,
            gwsEnv={k: v for k, v in sorted(os.environ.items()) if k.startswith('GWS_')},
            inContainer=gws.c.env.GWS_IN_CONTAINER,
            uwsgi='',
            userName=ui['pw_name'],
            groupName=ui['gr_name'],
            homeDir=ui['pw_dir'],
            mapproxyConfig='',
            mapproxyPid=pid_paths['mapproxy'],
            mapproxySocket=f'{gws.c.TMP_DIR}/mapproxy.uwsgi.sock',
            nginxConfig='',
            nginxPid=pid_paths['nginx'],
            spoolConfig='',
            spoolPid=pid_paths['spool'],
            spoolSocket=f'{gws.c.TMP_DIR}/spool.uwsgi.sock',
            syslogConfig='',
            syslogPid='',
            webConfig='',
            webPid=pid_paths['web'],
            webSocket=f'{gws.c.TMP_DIR}/web.uwsgi.sock',
        )

        if args.inContainer:
            args.syslogPid = f'{gws.c.PIDS_DIR}/rsyslogd.pid'
            args.syslogConfig = self._create_config('server.rsyslog_config', f'{target_dir}/syslog.conf', args)

        if self.cfg('withWeb'):
            args.uwsgi = 'web'
            args.webConfig = self._create_config('server.uwsgi_config', f'{target_dir}/uwsgi_web.ini', args)

        if self.cfg('withMapproxy') and gws.u.is_file(gws.gis.mpx.config.CONFIG_PATH):
            args.uwsgi = 'mapproxy'
            args.mapproxyConfig = self._create_config('server.uwsgi_config', f'{target_dir}/uwsgi_mapproxy.ini', args)

        if self.cfg('withSpool'):
            args.uwsgi = 'spool'
            args.spoolConfig = self._create_config('server.uwsgi_config', f'{target_dir}/uwsgi_spool.ini', args)

        args.nginxConfig = self._create_config('server.nginx_config', f'{target_dir}/nginx.conf', args, True)

        self._create_config('server.start_script', script_path, args)

    def _create_config(self, subject: str, path: str, args: TemplateArgs, is_nginx=False) -> str:
        tpl = gws.u.require(self.root.app.templateMgr.find_template(subject, where=[self]))
        res = tpl.render(gws.TemplateRenderInput(args=args))

        text = str(res.content)
        if is_nginx:
            text = re.sub(r'\s*{\s*', ' {\n', text)
            text = re.sub(r'\s*}\s*', '\n}\n', text)

        lines = []
        indent = 0

        for line in text.split('\n'):
            line = re.sub(r'\s+', ' ', line.strip())
            if not line:
                continue
            if is_nginx:
                if line == '}':
                    indent -= 1
                lines.append((' ' * (indent * 4)) + line)
                if '{' in line:
                    indent += 1
            else:
                lines.append(line)

        text = '\n'.join(lines) + '\n'
        gws.u.write_file(path, text)
        return path
