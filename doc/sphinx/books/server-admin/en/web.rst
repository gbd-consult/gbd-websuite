Web Server
==========

Web content (html files, images, downloads) in GWS is processed by the bundled NGINX, a fast and reliable web-server.

You can configure multiple *sites* ("virtual hosts") in a single GBD WebSuite installation, each with its own hostname and document root.


Static files and assets
-----------------------

There are two types of web content: *static* resources and *assets*. Static resources are placed in a web root folder and are served as-is, without any processing. Assets are resources that require some processing on the server side before they can be served to the client.

Use-cases for static files are:

- plain, public html pages
- javascript and css files
- static images

Use-cases for assets are:

- any resource that requires authorization
- template-based html pages
- project-specific resources

GBD WebSuite only serves resources with known mime-types (determined by the file extension), the default set is ::

    .css .csv .gif .html .jpeg .jpg .js .json .pdf .png .svg .ttf .xml .zip

You can redefine this list on a per-site or per-project basis.

Rewrite rules
-------------

Assets are handled by the server command ``assetHttpGetPath``, which accepts the ``path`` parameter, and, optionally, a project unique id, so the final URL would be like this ::


    http://example.org/_?cmd=assetHttpGetPath&projectUid=myproject&path=somepage.html

The following rewrite rule ::

    {
        "match": "^/([a-z]+)/([a-z]+)$",
        "target": "_?cmd=assetHttpGetPath&projectUid=$1&path=$2.html"
    }


will transform this URL into simply ::

    http://example.org/myproject/somepage


The ``match`` is a reqular expression and the ``target`` is the final URL with ``{$n}`` placeholders, corresponding to capture groups in the regex. If the target starts with a scheme (e.g. ``http://``), the server performs a redirect instead of rewriting.


Website configuration
---------------------

A website configuration must include a hostname (hostname ``*`` marks the default site), document root and assets configurations, error page template, and, optionally, a set of URL rewriting rules. ::


    {

        ## hostname

        "host": "example.org",

        ## static document root

        "root": {

            ## absolute container path to the root directory

            "dir": "/data/www-root",

            ## allowed file extensions(in addition to the default list)

            "allowMime": [".xls", ".doc"],

            ## disabled file extensions(from the default list)

            "denyMime": [".xml", ".json"],

        },

        ## assets root

        "assets": {

            ## absolute container path to the site assets directory

            "dir": "/data/www-assets",

        },

        ## error page template

        "errorPage": {
            "type": "html",
            "path": "/data/error/template.html"
        },

        ## rewrite rules

        "rewrite": [

            {
                "match": "^/$",
                "target": "_?cmd=assetHttpGetPath&path=root-page.html"
            },
            {
                "match": "^/hello/([a-z]+)$",
                "target": "_?cmd=assetHttpGetPath&projectUid=hello_project&path=$1.html"
            }
        ]



Project assets
--------------

Each GBD WebSuite project can have its own assets root configured. When the GBD WebSuite client requests an asset without a project uid, like ::

    http://example.org/_?cmd=assetHttpGetPath&path=somepage.html

then the asset is looked for in the site assets directory. If a request comes with a project uid ::

    http://example.org/_?cmd=assetHttpGetPath&projectUid=myproject&path=somepage.html

then the asset is first looked for in project assets, if it's not found, the site assets directory is used as a fallback.

HTML Templates
--------------


Gws uses its own templating engine, which supports the following basic commands:

TABLE
   ``@if <condition> ... @end`` ~ Check a condition
   ``@each <object> as <key>, <value> ... @end`` ~ Iterate a key-value object
   ``@include <path>`` ~ Include another template
/TABLE

Property values can be inserted with an ``{object.property}`` construct, with optional filters, e.g. ``{{object.property | html}}``.

Here's an example of a feature formatting template ::

    @if feature.category
        <p class="head">{feature.category | html}</p>
    @end

    @if feature.title
        <p class="head2">{feature.title | html}</p>
    @end

    <table><tbody>

        @each feature.attributes as name, value
            <tr>
                <th>{name | html}</th>
                <td>{value | html | nl2br | linkify(target="_blank", cut=30)}</td>
            </tr>
        @end

    </tbody></table>


Error page template
-------------------

An error page template has access to the error code in the ``error`` variable. You can the ``@if`` command to provide different content, depending on the error ::


    <h1>Error!</h1>

    @if error == 404
        Resource not found
    @elif error == 403
        Access denied
    @else
        Error {error} has occured
    @end

SSL configuration
-----------------

SSL can configured under ``web.ssl``. You have to provide paths (as visible in the container) to your certificate bundle and the private key. SSL configuration is applied to the whole server, not to individually configured sites. With SSL, your ``web`` configuration would look like this ::

    "web": {
        "sites": [
            {
                "host": "..."
                // site configuration as described above
            },
            {
                "host": "..."
                // another site configuration
            }
            ...
        ],
        "ssl": {
            "crt": "/path/to/your-certificate-bundle.crt",
            "key": "/path/to/your-private-key.crt"
        }
    }

If your certificate consist of separate files (e.g. domain certificate, intermediate and root certificates) you can create the bundle by concatenating them together ::

    cat domain.crt intermediate.crt root.crt > bundle.crt
