import grp
import os
import pwd

import gws
import gws.base.web
import gws.config
import gws.config.util
import gws.gis.mpx.config
import gws.lib.osx

from . import core

PID_PATHS = {
    'web': f'{gws.c.PIDS_DIR}/web.uwsgi.pid',
    'spool': f'{gws.c.PIDS_DIR}/spool.uwsgi.pid',
    'mapproxy': f'{gws.c.PIDS_DIR}/mapproxy.uwsgi.pid',
    'nginx': f'{gws.c.PIDS_DIR}/nginx.pid',
}

_DEFAULT_BASE_TIMEOUT = 60
_DEFAULT_SPOOL_TIMEOUT = 300

_DIR = gws.u.dirname(__file__)

_DEFAULT_TEMPLATES = [
    gws.Config(type='text', subject='server.rsyslog_config', path=f'{_DIR}/templates/rsyslog_config.cx.txt'),
    gws.Config(type='text', subject='server.nginx_config', path=f'{_DIR}/templates/nginx_config.cx.txt'),
    gws.Config(type='text', subject='server.uwsgi_config', path=f'{_DIR}/templates/uwsgi_config.cx.txt'),
]


class Object(gws.ServerManager):
    def configure(self):
        self.config = self.config or gws.Data()
        self.configure_defaults()
        self.configure_environment()
        self.configure_templates()

    def configure_defaults(self):
        self.config.mapproxy = self.config.mapproxy or self.root.specs.read({}, 'gws.server.core.MapproxyConfig')
        self.config.monitor = self.config.monitor or self.root.specs.read({}, 'gws.server.core.MonitorConfig')
        self.config.log = self.config.log or self.root.specs.read({}, 'gws.server.core.LogConfig')
        self.config.qgis = self.config.qgis or self.root.specs.read({}, 'gws.server.core.QgisConfig')
        self.config.spool = self.config.spool or self.root.specs.read({}, 'gws.server.core.SpoolConfig')
        self.config.web = self.config.web or self.root.specs.read({}, 'gws.server.core.WebConfig')

    def configure_environment(self):
        p = gws.env.GWS_LOG_LEVEL
        if p:
            self.config.log.level = p
        p = gws.env.GWS_WEB_WORKERS
        if p:
            self.config.web.workers = int(p)
        p = gws.env.GWS_SPOOL_WORKERS
        if p:
            self.config.spool.workers = int(p)

    def configure_templates(self):
        gws.config.util.configure_templates(self, extra=_DEFAULT_TEMPLATES)

    def create_server_configs(self, target_dir, script_path):
        args = core.ConfigTemplateArgs(
            root=self.root,
            inContainer=gws.u.is_file('/.dockerenv'),
            userName=pwd.getpwuid(gws.c.UID).pw_name,
            groupName=grp.getgrgid(gws.c.GID).gr_name,
            gwsEnv={k: v for k, v in sorted(os.environ.items()) if k.startswith('GWS_')},
            mapproxyPid=PID_PATHS['mapproxy'],
            mapproxySocket=f'{gws.c.TMP_DIR}/mapproxy.uwsgi.sock',
            nginxPid=PID_PATHS['nginx'],
            spoolPid=PID_PATHS['spool'],
            spoolSocket=f'{gws.c.TMP_DIR}/spool.uwsgi.sock',
            webPid=PID_PATHS['web'],
            webSocket=f'{gws.c.TMP_DIR}/web.uwsgi.sock',
        )

        commands = []

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

        path = self._create_config('server.nginx_config', f'{target_dir}/nginx.conf', args)
        commands.append(f'exec nginx -c {path}')

        gws.u.write_file(script_path, '\n'.join(commands) + '\n')

    def _create_config(self, subject: str, path: str, args: core.ConfigTemplateArgs) -> str:
        tpl = self.root.app.templateMgr.find_template(self, subject=subject)
        res = tpl.render(gws.TemplateRenderInput(args=args))

        lines = []
        for s in str(res.content).strip().split('\n'):
            s = s.strip()
            if s:
                lines.append(s)
        text = '\n'.join(lines) + '\n'

        gws.u.write_file(path, text)
        return path
