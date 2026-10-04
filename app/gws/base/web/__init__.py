"""Web application, web site settings and assets.

This package contains the WSGI application that serves all web requests, the
web site configuration and the ``web`` action for pages, assets and files.

Submodules:

- ``wsgi_main``: the WSGI entry point loaded by uWSGI.
- ``wsgi_app``: the WSGI application. It loads the configuration on the first
  request, then creates a requester for each request, runs the middleware and
  dispatches the command to an action.
- ``wsgi``: the requester (``gws.WebRequester``) and responder
  (``gws.WebResponder``) implementations, based on Werkzeug.
- ``error``: HTTP exceptions, and the conversion of GWS errors to HTTP errors.
- ``manager``: the web manager (``gws.WebManager``), configured in the ``web``
  section of the application; it creates the web site. The ``ssl`` option of
  that section enables SSL for the site.
- ``site``: the web site (``gws.WebSite``): host names, SSL, CORS, security
  headers, static and assets directories, URL rewrite rules and URL generation.
- ``action``: the ``web`` action, which serves pages, assets, system assets
  (client scripts and styles), downloads and files stored in model fields.

Requests
--------

The server handles requests to ``/_`` and ``/_/<command>``. Parameters of GET
requests are taken from the query string or from the path, in the form
``/_/<command>/<name1>/<value1>/<name2>/<value2>``. POST requests with a JSON or
MessagePack body are API requests; the response is encoded in the format of the
``Accept`` header, or else in the request format. Errors are converted to HTTP
errors; for API requests they are returned as a structured response, otherwise
the ``application.error`` template is rendered, if there is one.

The rewrite rules of the site map incoming URLs to command URLs, and reversed
rules map generated command URLs back to readable ones. By default, ``/`` is the
application home page (``webPage`` with ``name=home``) and ``/project/<uid>``
is the project page (``webPage`` with ``name=project``). The default rules are
added unless ``withDefaultRewriteRules`` is false or a configured rule has the
same pattern. Relative rewrite targets are made absolute.

Assets
------

An asset is a file located in a global or project-specific assets directory.
If not configured, the global assets directory is ``/data/assets``, if it exists,
and the static root of the site is ``/data/web``, or a temporary directory if it
does not exist.
To access a project asset, the user must be allowed to use the project. When
the ``web`` action receives a ``webAsset`` request with a ``path`` argument, it
first checks the project assets directory, then the global one.

If the file is found and its extension is one of
``gws.base.template.manager.TEMPLATE_TYPES``, a template is created from it on
the fly and rendered with ``gws.base.web.action.TemplateArgs``. The response of
the template is passed back to the user. Other files are returned as is if their
MIME type passes the ``allowMime`` and ``denyMime`` filters of the directory;
without ``allowMime``, only common web types (HTML, CSS, JavaScript, images, PDF,
JSON, XML and similar) are served.

Example::

    {
        web.site {
            hostnames [ "maps.example.com" ]
            assets { dir "/data/assets" }
            rewriteRules [
                { pattern "^/maps/([a-z0-9_]+)$" target "/_/webPage/name/project/projectUid/$1" }
            ]
        }
        actions [
            { type "web" access "allow all" }
        ]
    }
"""

from . import error, manager, site, wsgi
