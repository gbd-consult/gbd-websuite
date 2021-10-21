import os

VERSION = '8.0.0'

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

CONFIG_DIR = VAR_DIR + '/config'
GLOBALS_DIR = TMP_DIR + '/globals'
LEGEND_CACHE_DIR = VAR_DIR + '/cache/legend'
LOCKS_DIR = TMP_DIR + '/lock'
LOG_DIR = VAR_DIR + '/log'
MAPPROXY_CACHE_DIR = VAR_DIR + '/cache/mpx'
MISC_DIR = VAR_DIR + '/misc'
NET_CACHE_DIR = VAR_DIR + '/cache/net'
OBJECT_CACHE_DIR = VAR_DIR + '/cache/object'
PRINT_DIR = TMP_DIR + '/print'
SERVER_DIR = VAR_DIR + '/server'
SPOOL_DIR = TMP_DIR + '/spool'
WEB_CACHE_DIR = TMP_DIR + '/webcache'

SERVER_ENDPOINT = '/_'

EPSG_3857 = 'EPSG:3857'
EPSG_4326 = 'EPSG:4326'

ROLE_ADMIN = 'admin'
ROLE_USER = 'user'
ROLE_GUEST = 'guest'
ROLE_ALL = 'all'