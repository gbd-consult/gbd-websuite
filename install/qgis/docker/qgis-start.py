"""Prepare the qgis server, uwsgi and nginx."""

import os
import urllib.parse

USER_UID = 1000
USER_GID = 1000

QGIS_DEBUG = os.getenv('QGIS_DEBUG', '0')
QGIS_WORKERS = os.getenv('QGIS_WORKERS', 1)
SVG_PATHS = os.getenv('SVG_PATHS', '')
TIMEOUT = os.getenv('TIMEOUT', '60')
HTTP_PROXY = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY')
PGSERVICEFILE = os.getenv('PGSERVICEFILE', '')


##


def _from_env(default):
    return lambda k: os.getenv(k, default)


class QgisEnv:
    QGIS_PREFIX_PATH = '/usr'

    GDAL_DEFAULT_WMS_CACHE_PATH = '/qgis/cache/gdal'
    GDAL_FIX_ESRI_WKT = 'GEOGCS'

    QGIS_OPTIONS_PATH = '/qgis/profiles/default'
    QGIS_GLOBAL_SETTINGS_FILE = '/qgis/profiles/default/QGIS/QGIS3.ini'
    QGIS_CUSTOM_CONFIG_PATH = '/qgis'

    # qgis server:
    # https://docs.qgis.org/3.28/en/docs/server_manual/config.html

    QGIS_PLUGINPATH = ''
    QGIS_PROJECT_FILE = ''

    QGIS_SERVER_ALLOWED_EXTRA_SQL_TOKENS = _from_env('')
    QGIS_SERVER_API_RESOURCES_DIRECTORY = ''
    QGIS_SERVER_API_WFS3_MAX_LIMIT = ''
    QGIS_SERVER_CACHE_DIRECTORY = '/qgis/cache/server'
    QGIS_SERVER_CACHE_SIZE = _from_env('10000000')
    QGIS_SERVER_DISABLE_GETPRINT = ''
    QGIS_SERVER_FORCE_READONLY_LAYERS = _from_env('1')
    QGIS_SERVER_IGNORE_BAD_LAYERS = _from_env('1')
    QGIS_SERVER_LANDING_PAGE_PREFIX = ''
    QGIS_SERVER_LANDING_PAGE_PROJECTS_DIRECTORIES = ''
    QGIS_SERVER_LANDING_PAGE_PROJECTS_PG_CONNECTIONS = ''
    QGIS_SERVER_LOG_FILE = ''
    QGIS_SERVER_LOG_LEVEL = _from_env('2')
    QGIS_SERVER_LOG_PROFILE = _from_env('0')
    QGIS_SERVER_LOG_STDERR = 0
    QGIS_SERVER_MAX_THREADS = 0
    QGIS_SERVER_OVERRIDE_SYSTEM_LOCALE = _from_env('')
    QGIS_SERVER_PARALLEL_RENDERING = 0
    QGIS_SERVER_PROJECT_CACHE_CHECK_INTERVAL = _from_env('')
    QGIS_SERVER_PROJECT_CACHE_STRATEGY = _from_env('filesystem')
    QGIS_SERVER_SERVICE_URL = ''
    QGIS_SERVER_SHOW_GROUP_SEPARATOR = _from_env('')
    QGIS_SERVER_TRUST_LAYER_METADATA = _from_env(0)
    QGIS_SERVER_WCS_SERVICE_URL = ''
    QGIS_SERVER_WFS_SERVICE_URL = ''
    QGIS_SERVER_WMS_MAX_HEIGHT = ''
    QGIS_SERVER_WMS_MAX_WIDTH = ''
    QGIS_SERVER_WMS_SERVICE_URL = ''
    QGIS_SERVER_WMTS_SERVICE_URL = ''


qgis_env = f'export QGIS_DEBUG={QGIS_DEBUG}\n'

for key, val in vars(QgisEnv).items():
    if key.startswith('_'):
        continue
    if callable(val):
        val = val(key)
    if val:
        qgis_env += f'export {key}={val}\n'

##

_ = os.system

_('mkdir -p /qgis/profiles/default/QGIS')
_('mkdir -p /qgis/profiles/profiles/default/QGIS')
_('mkdir -p /qgis/cache/gdal')
_('mkdir -p /qgis/cache/network')
_('mkdir -p /qgis/cache/server')
_(f'chown -R {USER_UID}:{USER_GID} /qgis')

_('mkdir -p /var/run')
_('chmod 777 /var/run')
_(f'chown -R {USER_UID}:{USER_GID} /var/run')

##

svg_paths = '/usr/share/qgis/svg,/usr/share/alkisplugin/svg'
if SVG_PATHS:
    svg_paths += ',' + SVG_PATHS

proxy = ''
if HTTP_PROXY:
    p = urllib.parse.urlsplit(HTTP_PROXY)
    proxy = f'''
[proxy]
proxyEnabled=true
proxyType=HttpProxy
proxyHost={p.hostname}
proxyPort={p.port}
proxyUser={p.username}
proxyPassword={p.password}
    '''

qgis_ini = fr"""
[cache]
directory=/qgis/cache/network
size=@Variant(\0\0\0\x81\0\0\0\0\0@\0\0)

[qgis]
symbolsListGroupsIndex=0

[svg]
searchPathsForSVG={svg_paths}

{proxy}
"""

##

uwsgi_ini = fr"""
[uwsgi]
uid = {USER_UID}
gid = {USER_GID}
chmod-socket = 666
fastcgi-socket = /var/run/uwsgi.sock
daemonize = true
die-on-term = true
logger = syslog:QGIS,local6
master = true
pidfile = /var/run/uwsgi.pid
processes = {QGIS_WORKERS}
reload-mercy = 5
threads = 0
vacuum = true
worker-exec = /usr/bin/qgis_mapserv.fcgi
worker-reload-mercy = 5
harakiri = {TIMEOUT}
"""

##
custom_fcgi_params = {
        'PGSERVICEFILE': PGSERVICEFILE
}
def fcgi_params (params):
    return '\n'.join([f'fastcgi_param {k} {v};' for k,v in params.items() if v])

nginx_conf = fr"""
user gws;
worker_processes auto;
pid /var/run/nginx.pid;
daemon off;
error_log syslog:server=unix:/dev/log,nohostname,tag=NGINX warn;

events {{
    worker_connections 512;
}}

http {{
    access_log syslog:server=unix:/dev/log,nohostname,tag=NGINX;
    server {{
        listen 80;
        location / {{
            gzip off;
            fastcgi_pass unix:/var/run/uwsgi.sock;
            fastcgi_read_timeout {TIMEOUT}s;
            add_header 'Access-Control-Allow-Origin' *;
            include /etc/nginx/fastcgi_params;
            {fcgi_params(custom_fcgi_params)}
            
            # replace mapproxy forward params (e.g. LAYERS__gws) with their real names
            
            if ($args ~* (.*?)__gws(.*)) {{
                set $args $1$2;
            }}
            if ($args ~* (.*?)__gws(.*)) {{
                set $args $1$2;
            }}
            if ($args ~* (.*?)__gws(.*)) {{
                set $args $1$2;
            }}
        }}
    }}
}}
"""

##

# silence some warnings unless debugging

silence = '''
# 'QFont::setPointSize: Point size must be greater than 0'
:msg, contains, "QFont::setPointSizeF" stop
'''

if QGIS_DEBUG != '0':
    silence = ''

rsyslogd_conf = fr"""
module(
    load="imuxsock"
    SysSock.UsePIDFromSystem="on"
)

template(name="gws" type="list") {{
    property(name="timestamp" dateFormat="mysql")
    constant(value=" ")
    property(name="syslogtag")
    property(name="msg" spifno1stsp="on" )
    property(name="msg" droplastlf="on" )
    constant(value="\\n")
}}

module(
    load="builtin:omfile" 
    Template="gws"
)

{silence}

*.* /dev/stdout
"""


##

def write(p, s):
    with open(p, 'wt') as fp:
        fp.write(s.strip() + '\n')
    _(f'chmod 666 {p}')


write('/qgis/profiles/default/QGIS/QGIS3.ini', qgis_ini)
write('/uwsgi.ini', uwsgi_ini)
write('/nginx.conf', nginx_conf)
write('/rsyslogd.conf', rsyslogd_conf)

_(f'chown -R {USER_UID}:{USER_GID} /qgis')

qgis_start_configured = f"""
#!/bin/bash

export DISPLAY=:99
export LC_ALL=C.UTF-8
export XDG_RUNTIME_DIR=/tmp/xdg

{qgis_env}

rm -fr /tmp/*

XVFB=/usr/bin/Xvfb
XVFBARGS='-dpi 96 -screen 0 1024x768x24 -ac +extension GLX +render -noreset -nolisten tcp'

until start-stop-daemon --status --exec $XVFB; do
    echo 'waiting for xvfb...'
    start-stop-daemon --start --background --exec $XVFB --oknodo -- $DISPLAY $XVFBARGS
    sleep 0.5
done

rsyslogd -i /var/run/rsyslogd.pid -f /rsyslogd.conf
uwsgi /uwsgi.ini
exec nginx -c /nginx.conf
"""

write('/qgis-start-configured', qgis_start_configured)
_('chmod 777 /qgis-start-configured')

print(f'\nQGIS ENVIRONMENT:\n\n{qgis_env}')
