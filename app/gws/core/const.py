import os

VERSION='5.2.16'

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

MAPPROXY_CACHE_DIR = VAR_DIR + '/cache/mpx'
OBJECT_CACHE_DIR = VAR_DIR + '/cache/object'
NET_CACHE_DIR = VAR_DIR + '/cache/net'
WEB_CACHE_DIR = TMP_DIR + '/webcache'

LOG_DIR = VAR_DIR + '/log'
MISC_DIR = VAR_DIR + '/misc'
CONFIG_DIR = VAR_DIR + '/config'
SERVER_DIR = VAR_DIR + '/server'

PRINT_DIR = TMP_DIR + '/print'
SPOOL_DIR = TMP_DIR + '/spool'

SERVER_ENDPOINT = '/_'

# from uwsgi
SPOOL_OK = -2
SPOOL_RETRY = -1
SPOOL_IGNORE = 0

REMOTE_ADMIN_PASSWD_FILE = VAR_DIR + '/.remoteadmin'

EPSG_3857 = 'EPSG:3857'
EPSG_4326 = 'EPSG:4326'
