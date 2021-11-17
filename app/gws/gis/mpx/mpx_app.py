# https://mapproxy.org/docs/1.11.0/deployment.html?highlight=make_wsgi_app#server-script

import logging
import os

from mapproxy.wsgiapp import make_wsgi_app

import gws

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

cfg = gws.CONFIG_DIR + '/mapproxy.yaml'

if os.path.isfile(cfg):
    application = make_wsgi_app(cfg)

# import os, sys
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# from util import AnnotationFilter
# application = AnnotationFilter(application)
