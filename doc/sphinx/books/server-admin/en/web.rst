Web Server
==========

Web content (html files, images, downloads) in GWS is processed by the bundled NGINX, a fast and reliable web-server.

You can configure multiple *sites* ("virtual hosts") in a single GBD WebSuite installation, each with its own hostname and document root.


Static files and assets
-----------------------

There are two types of web content: *static* resources and *assets*. Static resources are placed in a web root folder and are served as-is, without any processing. Assets are resources that require some processing on the server side before they can be served to the client.

Use-cases for static files are:

- plain, public html pages
- javascript and css files (including the GBD WebSuite Client material)
- static images

Use-cases for assets are:

- any resource that requires authorization
- template-based html pages (GBD WebSuite uses `mako <https:--www.makotemplates.org/>`_ for templating)
- project-specific resources

GBD WebSuite only serves resources with known mime-types (determined by the file extension), the default set is ::

    .css .csv .gif .html .jpeg .jpg .js .json .pdf .png .svg .ttf .xml .zip

You can redefine this list on a per-site or per-project basis.

Rewrite rules
-------------

Assets are handled by the server command ``assetHttpGetPath``, which accepts the ``path`` parameter, and, optionally, a project unique id, so the final URL would be like this ::


    http://example.org/_?cmd=assetHttpGetPath&projectUid=myproject&path=somepage.mako

The following rewrite rule ::

    {
        "match": "^/([a-z]+)/([a-z]+)$",
        "target": "_?cmd=assetHttpGetPath&projectUid=$1&path=$2.mako"
    }


will transform this URL into simply ::

    http://example.org/myproject/somepage


The ``match`` is a reqular expression and the ``target`` is the final URL with ``{$n}`` placeholders, corresponding to capture groups in the regex. If the target starts with a scheme (e.g. ``http://``), the server performs a redirect instead of rewriting.


Website configuration
---------------------

A website configuration must include a hostname (hostname ``*`` marks the default site), document root and assets configurations, and, optionally, a set of URL rewriting rules ::


    {

        ## hostname


        "host": "example.org",

        ## static document root

        "root": {

            ## absolute path to the root directory

            "dir": "/example/www-root",

            ## allowed file extensions(in addition to the default list)

            "allowMime": [".xls", ".doc"],

            ## disabled file extensions(from the default list)

            "denyMime": [".xml", ".json"],

        },

        ## assets root

        "assets": {

            ## absolute path to the site assets directory

            "dir": "/example/www-assets",

        },

        ## rewrite rules

        "rewrite": [

            {
                "match": "^/$",
                "target": "_?cmd=assetHttpGetPath&path=root-page.mako"
            },
            {
                "match": "^/hello/([a-z]+)$",
                "target": "_?cmd=assetHttpGetPath&projectUid=hello_project&path=$1.mako"
            }
        ]


Project assets
--------------

Each GBD WebSuite project can have its own assets root configured. When the GBD WebSuite client requests an asset without a project uid, like ::

    http://example.org/_?cmd=assetHttpGetPath&path=somepage.mako

then the asset is looked for in the site assets directory. If a request comes with a project uid ::

    http://example.org/_?cmd=assetHttpGetPath&projectUid=myproject&path=somepage.mako

then the asset is first looked for in project assets, if it's not found, the site assets directory is used as a fallback.





