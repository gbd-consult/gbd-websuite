# https://mapproxy.org/docs/1.11.0/deployment.html?highlight=make_wsgi_app#server-script

import gws
import gws.core.error

import mapproxy.wsgiapp
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# from mapproxy.wsgiapp, adapted to our needs

def make_wsgi_app(services_conf=None, debug=False, ignore_config_warnings=True, reloader=False):
    conf = mapproxy.wsgiapp.load_configuration(mapproxy_conf=services_conf, ignore_warnings=ignore_config_warnings)
    services = conf.configured_services()

    config_files = conf.config_files()
    conf.base_config.debug_mode = True

    app = mapproxy.wsgiapp.MapProxyApp(services, conf.base_config)

    app.config_files = config_files
    return app


mapproxy_app = make_wsgi_app(gws.CONFIG_DIR + '/mapproxy.yaml')


def init():
    pass


def application(environ, start_response):
    try:
        return mapproxy_app(environ, start_response)
    except Exception as exc:
        gws.log.exception()
        headers = [('Content-type', 'text/plain')]
        start_response('500', headers)
        return [('Exception\n' + repr(exc))]

# import os, sys
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# from util import AnnotationFilter
# application = AnnotationFilter(application)
