import os

APP_DIR = '/gws-app'
VAR_DIR = '/gws-var'
TMP_DIR = '/tmp/gws'

UID = 1000
GID = 1000

APP_DIR = os.getenv('GWS_APP_DIR', APP_DIR)
VAR_DIR = os.getenv('GWS_VAR_DIR', VAR_DIR)
TMP_DIR = os.getenv('GWS_TMP_DIR', TMP_DIR)

UID = int(os.getenv('GWS_UID', UID))
GID = int(os.getenv('GWS_GID', GID))

CACHE_DIR = f'{VAR_DIR}/cache'
LEGEND_CACHE_DIR = f'{CACHE_DIR}/legend'
MAPPROXY_CACHE_DIR = f'{CACHE_DIR}/mpx'
NET_CACHE_DIR = f'{CACHE_DIR}/net'
OBJECT_CACHE_DIR = f'{CACHE_DIR}/object'

CONFIG_DIR = f'{VAR_DIR}/config'
MISC_DIR = f'{VAR_DIR}/misc'
SERVER_DIR = f'{VAR_DIR}/server'
PRINT_DIR = f'{VAR_DIR}/print'

FASTCACHE_DIR = f'{TMP_DIR}/fastcache'
LOCKS_DIR = f'{TMP_DIR}/locks'
GLOBALS_DIR = f'{TMP_DIR}/globals'
SPOOL_DIR = f'{TMP_DIR}/spool'

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
