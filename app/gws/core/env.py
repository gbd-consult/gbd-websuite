"""Environment variables that override configuration values."""

import os

GWS_APP_DIR = os.getenv('GWS_APP_DIR')
"""Application directory."""

GWS_VAR_DIR = os.getenv('GWS_VAR_DIR')
"""Var data directory."""

GWS_TMP_DIR = os.getenv('GWS_TMP_DIR')
"""Temporary directory."""

GWS_UID = os.getenv('GWS_UID')
"""Server user uid."""

GWS_GID = os.getenv('GWS_GID')
"""Server user group id."""

GWS_CONFIG = os.getenv('GWS_CONFIG')
"""Path to the config file."""

GWS_MANIFEST = os.getenv('GWS_MANIFEST')
"""Path to the manifest file."""

GWS_LOG_LEVEL = os.getenv('GWS_LOG_LEVEL')
"""Log level."""

GWS_WEB_WORKERS = os.getenv('GWS_WEB_WORKERS')
"""Number of web workers to start."""

GWS_SPOOL_WORKERS = os.getenv('GWS_SPOOL_WORKERS')
"""Number of spool workers to start."""

GWS_ALKIS_GID_VERSION = os.getenv('GWS_ALKIS_GID_VERSION')
"""GeoInfoDok version of the ALKIS source data, ``6`` or ``7`` (default)."""

GWS_IN_CONTAINER = os.path.isfile('/.dockerenv')
"""True if we're running in a container."""

GWS_IN_TEST = (os.getenv('GWS_IN_TEST') == '1') or (os.getenv('PYTEST_CURRENT_TEST') is not None)
"""True if we're running tests."""

GWS_TEST_INI = os.getenv('GWS_TEST_INI')
"""Location of the test.ini file."""

GWS_TEST_DIR = os.getenv('GWS_TEST_DIR')
"""Test base directory."""
