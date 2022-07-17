"""Prepare the qgis server, uwsgi and nginx."""

import os

QGIS_DEBUG = int(os.environ.get('QGIS_DEBUG', 0))
QGIS_WORKERS = os.environ.get('QGIS_WORKERS', 1)
SVG_PATHS = os.environ.get('SVG_PATHS', '')
ALLOW_IPS = os.environ.get('ALLOW_IPS', 'all')
TIMEOUT = os.environ.get('TIMEOUT', '60')

####

os.makedirs('/qgis/profiles/default/QGIS', exist_ok=True)
os.makedirs('/qgis/profiles/profiles/default/QGIS', exist_ok=True)
os.makedirs('/qgis/cache/gdal', exist_ok=True)
os.makedirs('/qgis/cache/network', exist_ok=True)
os.makedirs('/qgis/cache/server', exist_ok=True)

s = '/usr/share/qgis/svg,/usr/share/alkisplugin/svg'
if SVG_PATHS:
    s += ',' + SVG_PATHS

qgis_ini = fr"""\
[cache]
directory=/qgis/cache/network
size=@Variant(\0\0\0\x81\0\0\0\0\0@\0\0)

[qgis]
symbolsListGroupsIndex=0

[svg]
searchPathsForSVG={s}
"""

uwsgi_ini = fr"""
[uwsgi]
uid = 33
gid = 33
chmod-socket = 666
fastcgi-socket = /tmp/uwsgi.sock
daemonize = true
logger = syslog:QGIS,local6
master = true
pidfile = /tmp/uwsgi.pid
processes = {QGIS_WORKERS}
reload-mercy = 5
threads = 0
vacuum = true
worker-exec = /usr/bin/qgis_mapserv.fcgi
worker-reload-mercy = 5
harakiri = {TIMEOUT}

env = GDAL_DEFAULT_WMS_CACHE_PATH=/qgis/cache/gdal
env = GDAL_FIX_ESRI_WKT=GEOGCS

# https://docs.qgis.org/3.22/en/docs/server_manual/config.html

env = QGIS_OPTIONS_PATH=/qgis/profiles/default
env = QGIS_SERVER_CACHE_DIRECTORY=/qgis/cache/server
env = QGIS_SERVER_CACHE_SIZE=10000000
env = QGIS_SERVER_IGNORE_BAD_LAYERS=1
env = QGIS_SERVER_LOG_LEVEL=0
env = QGIS_SERVER_LOG_PROFILE=0
env = QGIS_SERVER_PARALLEL_RENDERING=0

env = QGIS_DEBUG={QGIS_DEBUG}
env = QGIS_PREFIX_PATH=/usr
"""

allow = '\n'.join('allow ' + ip.strip() + ';' for ip in ALLOW_IPS.split(','))

nginx_conf = fr"""
user www-data;
worker_processes auto;
pid /tmp/nginx.pid;
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
            {allow}
            deny all;
            gzip off;
            fastcgi_pass unix:/tmp/uwsgi.sock;
            fastcgi_read_timeout {TIMEOUT}s;
            add_header 'Access-Control-Allow-Origin' *;
            include /etc/nginx/fastcgi_params;
        }}
    }}
}}
"""

rsyslogd_conf = fr"""
module(
    load="imuxsock"
    SysSock.UsePIDFromSystem="on"
)

template(name="gws" type="list") {{
    property(name="timestamp" dateFormat="rfc3339")
    constant(value=" ")
    property(name="syslogtag")
    constant(value=" ")
    property(name="msg" spifno1stsp="on" )
    property(name="msg" droplastlf="on" )
    constant(value="\\n")
}}

module(
    load="builtin:omfile" 
    Template="gws"
)

*.* /dev/stdout
"""


def write(p, s):
    with open(p, 'wt') as fp:
        fp.write(s)


write('/qgis/profiles/default/QGIS/QGIS3.ini', qgis_ini)
write('/uwsgi.ini', uwsgi_ini)
write('/nginx.conf', nginx_conf)
write('/rsyslogd.conf', rsyslogd_conf)

www_uid = 33
os.system(f'chown -R {www_uid}:{www_uid} /qgis')
