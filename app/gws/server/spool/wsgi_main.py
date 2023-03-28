import gws.server.spool.wsgi_app as wsgi_app

wsgi_app.init()
application = wsgi_app.application
