"""WSGI application for MapProxy integration with GWS.

This module provides a WSGI application that serves MapProxy requests.
It is based on the MapProxy WSGI application with adaptations for GWS.

See: https://mapproxy.org/docs/1.11.0/deployment.html?highlight=make_wsgi_app#server-script
"""

import yaml
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import gws

import mapproxy.wsgiapp
import logging

from . import config

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# from mapproxy.wsgiapp, adapted to our needs

def make_wsgi_app(services_conf: Optional[str] = None, debug: bool = False, 
                 ignore_config_warnings: bool = True, reloader: bool = False) -> mapproxy.wsgiapp.MapProxyApp:
    """Create a MapProxy WSGI application.

    This function creates a MapProxy WSGI application from a configuration file.
    It is adapted from mapproxy.wsgiapp.make_wsgi_app to suit GWS needs.

    Args:
        services_conf: Path to the MapProxy configuration file.
        debug: Whether to enable debug mode.
        ignore_config_warnings: Whether to ignore configuration warnings.
        reloader: Whether to enable the configuration reloader.

    Returns:
        A MapProxy WSGI application.
    """
    conf = mapproxy.wsgiapp.load_configuration(mapproxy_conf=services_conf, ignore_warnings=ignore_config_warnings)
    services = conf.configured_services()

    config_files = conf.config_files()
    conf.base_config.debug_mode = True

    app = mapproxy.wsgiapp.MapProxyApp(services, conf.base_config)

    app.config_files = config_files
    return app


mapproxy_app = make_wsgi_app(config.CONFIG_PATH)


def init() -> None:
    """Initialize the MapProxy application.

    This function reads the MapProxy configuration file and ensures that
    all required directories exist.
    """
    cfg = yaml.safe_load(gws.u.read_file(config.CONFIG_PATH))
    gws.u.ensure_dir(cfg['globals']['cache']['base_dir'])
    gws.u.ensure_dir(cfg['globals']['cache']['lock_dir'])
    gws.u.ensure_dir(cfg['globals']['cache']['tile_lock_dir'])


def application(environ: Dict[str, Any], start_response: Callable) -> List[bytes]:
    """WSGI application function for MapProxy.

    This function handles WSGI requests for MapProxy. It catches any exceptions
    that might occur and returns an error response instead of letting the
    exception propagate.

    Args:
        environ: The WSGI environment dictionary.
        start_response: The WSGI start_response callable.

    Returns:
        A list of bytes containing the response body.
    """
    try:
        return mapproxy_app(environ, start_response)
    except Exception as exc:
        gws.log.exception()
        headers = [('Content-type', 'text/plain')]
        start_response('500', headers)
        return [(f'Exception\n{repr(exc)}').encode('utf-8')]

# import os, sys
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# from util import AnnotationFilter
# application = AnnotationFilter(application)
