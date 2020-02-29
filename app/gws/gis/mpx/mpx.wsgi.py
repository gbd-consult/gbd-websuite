# https://mapproxy.org/docs/1.11.0/deployment.html?highlight=make_wsgi_app#server-script

import gws

from mapproxy.wsgiapp import make_wsgi_app
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

application = make_wsgi_app(gws.CONFIG_DIR + '/mapproxy.yaml')

# import os, sys
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# from util import AnnotationFilter
# application = AnnotationFilter(application)
