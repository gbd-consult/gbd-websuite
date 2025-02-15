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
    serverDir: str
    """Absolute path to app/server directory."""
    spoolPid: str
    """Spooler pid path."""
    spoolSocket: str
    """Spooler socket path."""
    uwsgi: str
    """uWSGI backend name."""
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
]


class Object(gws.ServerManager):
    def configure(self):
        self.configure_environment()
        self.configure_templates()

    def configure_environment(self):
        """Overwrite config values from the environment."""

        p = gws.env.GWS_LOG_LEVEL
        if p:
            cast(core.Config, self.config).log.level = p
        p = gws.env.GWS_WEB_WORKERS
        if p:
            cast(core.Config, self.config).web.workers = int(p)
        p = gws.env.GWS_SPOOL_WORKERS
        if p:
            cast(core.Config, self.config).spool.workers = int(p)

    def configure_templates(self):
        gws.config.util.configure_templates_for(self, extra=_DEFAULT_TEMPLATES)

    def create_server_configs(self, target_dir, script_path, pid_paths):
        ui = gws.lib.osx.user_info(gws.c.UID, gws.c.GID)
        args = TemplateArgs(
            root=self.root,
            inContainer=gws.c.env.GWS_IN_CONTAINER,
            userName=ui['pw_name'],
            groupName=ui['gr_name'],
            gwsEnv={k: v for k, v in sorted(os.environ.items()) if k.startswith('GWS_')},
            mapproxyPid=pid_paths['mapproxy'],
            mapproxySocket=f'{gws.c.TMP_DIR}/mapproxy.uwsgi.sock',
            nginxPid=pid_paths['nginx'],
            serverDir=_SERVER_DIR,
            spoolPid=pid_paths['spool'],
            spoolSocket=f'{gws.c.TMP_DIR}/spool.uwsgi.sock',
            webPid=pid_paths['web'],
            webSocket=f'{gws.c.TMP_DIR}/web.uwsgi.sock',
        )

        commands = []

        commands.append(f"export HOME={ui['pw_dir']}")

        if args.inContainer:
            path = self._create_config('server.rsyslog_config', f'{target_dir}/syslog.conf', args)
            commands.append(f'rsyslogd -i {gws.c.PIDS_DIR}/rsyslogd.pid -f {path}')

        if self.cfg('web.enabled'):
            args.uwsgi = 'web'
            path = self._create_config('server.uwsgi_config', f'{target_dir}/uwsgi_web.ini', args)
            commands.append(f'uwsgi --ini {path}')

        if self.cfg('mapproxy.enabled') and gws.u.is_file(gws.gis.mpx.config.CONFIG_PATH):
            args.uwsgi = 'mapproxy'
            path = self._create_config('server.uwsgi_config', f'{target_dir}/uwsgi_mapproxy.ini', args)
            commands.append(f'uwsgi --ini {path}')

        if self.cfg('spool.enabled'):
            args.uwsgi = 'spool'
            path = self._create_config('server.uwsgi_config', f'{target_dir}/uwsgi_spool.ini', args)
            commands.append(f'uwsgi --ini {path}')

        path = self._create_config('server.nginx_config', f'{target_dir}/nginx.conf', args, True)
        commands.append(f'exec nginx -c {path}')

        gws.u.write_file(script_path, '\n'.join(commands) + '\n')

    def _create_config(self, subject: str, path: str, args: TemplateArgs, is_nginx=False) -> str:
        tpl = self.root.app.templateMgr.find_template(subject, where=[self])
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
