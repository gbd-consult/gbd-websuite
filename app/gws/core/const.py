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
LEGEND_CACHE_DIR = VAR_DIR + '/cache/legend'
LOG_DIR = VAR_DIR + '/log'
MAPPROXY_CACHE_DIR = VAR_DIR + '/cache/mpx'
MISC_DIR = VAR_DIR + '/misc'
NET_CACHE_DIR = VAR_DIR + '/cache/net'
OBJECT_CACHE_DIR = VAR_DIR + '/cache/object'
SERVER_DIR = VAR_DIR + '/server'
WEB_CACHE_DIR = TMP_DIR + '/webcache'

LOCKS_DIR = TMP_DIR + '/locks'
GLOBALS_DIR = TMP_DIR + '/globals'
SPOOL_DIR = TMP_DIR + '/spool'
EPH_DIR = TMP_DIR + '/ephemeral'

SERVER_ENDPOINT = '/_'

ROLE_ADMIN = 'admin'
ROLE_USER = 'user'
ROLE_GUEST = 'guest'
ROLE_ALL = 'all'