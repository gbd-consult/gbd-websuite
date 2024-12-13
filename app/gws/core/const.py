from . import env

import os

APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../..')

VAR_DIR = env.GWS_VAR_DIR or '/gws-var'
TMP_DIR = env.GWS_TMP_DIR or '/tmp/gws'

UID = int(env.GWS_UID or 1000)
GID = int(env.GWS_GID or 1000)

CACHE_DIR = f'{VAR_DIR}/cache'
LEGEND_CACHE_DIR = f'{CACHE_DIR}/legend'
MAPPROXY_CACHE_DIR = f'{CACHE_DIR}/mpx'
NET_CACHE_DIR = f'{CACHE_DIR}/net'
OBJECT_CACHE_DIR = f'{CACHE_DIR}/object'

CONFIG_DIR = f'{VAR_DIR}/config'
MISC_DIR = f'{VAR_DIR}/misc'
SERVER_DIR = f'{VAR_DIR}/server'
QGIS_DIR = f'{VAR_DIR}/qgis'

FASTCACHE_DIR = f'{TMP_DIR}/fastcache'
PIDS_DIR = f'{TMP_DIR}/pids'
SPOOL_DIR = f'{TMP_DIR}/spool'
SPEC_DIR = f'{TMP_DIR}/spec'

TRANSIENT_DIR = f'{TMP_DIR}/transient'
LOCKS_DIR = f'{TRANSIENT_DIR}/locks'
GLOBALS_DIR = f'{TRANSIENT_DIR}/globals'
EPHEMERAL_DIR = f'{TRANSIENT_DIR}/ephemeral'

ALL_DIRS = [
    CACHE_DIR,
    LEGEND_CACHE_DIR,
    MAPPROXY_CACHE_DIR,
    NET_CACHE_DIR,
    OBJECT_CACHE_DIR,
    CONFIG_DIR,
    MISC_DIR,
    SERVER_DIR,
    QGIS_DIR,
    FASTCACHE_DIR,
    PIDS_DIR,
    SPOOL_DIR,
    SPEC_DIR,
    TRANSIENT_DIR,
    LOCKS_DIR,
    GLOBALS_DIR,
    EPHEMERAL_DIR,
]

SERVER_ENDPOINT = '/_'

ROLE_ADMIN = 'admin'
ROLE_USER = 'user'
ROLE_GUEST = 'guest'
ROLE_ALL = 'all'

ALLOW = 1
DENY = 0

PUBLIC = 'allow all'

JS_BUNDLE = "app.bundle.json"
JS_VENDOR_BUNDLE = 'vendor.bundle.js'
JS_UTIL_BUNDLE = 'util.bundle.js'
