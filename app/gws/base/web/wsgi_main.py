"""WSGI entry point for the web server."""

import gws.base.web.wsgi_app as wsgi_app

application = wsgi_app.application
