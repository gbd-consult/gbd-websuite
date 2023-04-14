"""Environment variables.

These variables, if set, override corresponding configuration values.
"""

import os

GWS_APP_DIR = os.getenv('GWS_APP_DIR')
"""application directory"""

GWS_VAR_DIR = os.getenv('GWS_VAR_DIR')
"""var data directory"""

GWS_TMP_DIR = os.getenv('GWS_TMP_DIR')
"""temporary directory"""

GWS_UID = os.getenv('GWS_UID')
"""server user uid"""

GWS_GID = os.getenv('GWS_GID')
"""server user group id"""

GWS_CONFIG = os.getenv('GWS_CONFIG')
"""path to the config file"""

GWS_MANIFEST = os.getenv('GWS_MANIFEST')
"""path to the manifest file"""

GWS_LOG_LEVEL = os.getenv('GWS_LOG_LEVEL')
"""log level"""

GWS_WEB_WORKERS = os.getenv('GWS_WEB_WORKERS')
"""number of web workers to start"""

GWS_SPOOL_WORKERS = os.getenv('GWS_SPOOL_WORKERS')
"""number of spool workers to start"""
