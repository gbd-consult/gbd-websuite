Projects
========

A *project* in GWS consists of a map, print templates and additional options. Besides that, you can override some application options for each project individually.

Project locations
-----------------

There are three ways to add projects to your GWS installation (you can also combine them):

TABLE
*projects* ~ configure projects directly in the main configuration
*projectPaths* ~ include specific project configurations
*projectDirs* ~ include all projects from directories
/TABLE

When using ``projectDirs``, the server searches the specified directories recursively and adds all files that end with ``config.py`` or ``config.json`` or ``config.yaml``.

Each project configuration file can contain a configuration for a single project, or an array of such configurations.

Project configurations
----------------------

A project configuration should contain at least ``uid`` (project unique id), ``title`` and ``map`` configurations. Some options, like ``access``, ``assets`` and ``client`` override respective application and website options.  ``printer`` describes project print templates (see :doc:`print`). Here's an example of a minimal project configuration ::

    {
        "title": "Hello",
        "uid": "project1",
        "map": {
            "view": {
                "extent": [554000, 6461000, 954000, 6861000],
                "scales": [1e3, 5e3, 1e4]
            },
            "layers": [
                {
                    "title": "OpenStreetMap",
                    "type": "client",
                    "kind": "osm"
                }
            ]
        }
    }



Maps
----

A *map* is basically a collection of *layers* (see :doc:`layers`). A map configuration can also contain Cache, Grid and View options that serve as fallback for layers that don't define them explicitly. The ``crs`` option must be valid EPSG CRS reference string. All layers in the project will be displayed in the CRS. Sources with different projections will be dynamically reprojected.

Multi-projects
--------------

A project configuration can also serve as a template for multiple projects. To set up a template, provide a regular expression in the ``multi`` option. The server enumerates all files on the server that match that expression, and creates a project config for each file found by replacing placeholders in other options' values. The placeholders are in the form ``{{$<number>}}`` where the number indicates a capturing group in the ``multi`` regex. ``{{$0}}`` is replaced with the complete path matched.

For example, this template will enumerate all QGIS projects in ``/data/qgis-maps`` and create a project with a qgis layer for each ``.qgs`` file, using the file name as a project title ::

        "multi": "/data/qgis-maps/(.+?).qgs$",
        "title": "Project {{$1}}",
        "map": {
            "layers": [
                {
                    "type": "qgis",
                    "path": "{{$0}}"
                }
            ]
        }

In addition to regex placeholders, following placeholders are supported

TABLE
``{{path}}`` ~ full path of the current file
``{{dirname}}`` ~ directory name of the current file
``{{filename}}`` ~ file name of the current file
``{{index}}`` ~ index of the current file in the list
/TABLE


Project HTML page
-----------------

To display your project in a web browser, you need an HTML page that should contain our Javascript Client (see :doc:`client`) and the project ID, so that the client knows which project to load. There must be a ``div`` element on the page with the class name ``gws``. This is where the client UI will be loaded. Otherwise, you can design your start page freely. Here's an example ::


    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8"/>
        <title>My First Project!</title>

        <!-- Load the Client, 2.3.1 is the version you're using -->
        <link rel="stylesheet" href="/gws-client/gws-light-2.3.1.css" type="text/css">
        <script src="/gws-client/gws-vendor-2.3.1.js"></script>
        <script src="/gws-client/gws-client-2.3.1.js"></script>

        <!-- Position the Client as you wish -->
        <style>
            .gws {
                position: fixed;
                left: 10px;
                top: 20px;
                right: 40px;
                bottom: 50px;
            }
        </style>

        <!-- Project uid, as defined in the config file -->
        <script>
            GWS_PROJECT_UID = "project1";
        </script>

        <!-- Your own css, if needed -->
        <link rel="stylesheet" href="/my-style.css" type="text/css">

        <!-- Your additional css/scripts and other resources -->

    </head>

    <body>
        <!-- This is where the Client will be loaded -->
        <div class="gws"></div>

        You can add more content here...
    </body>
    </html>

Place this file in your configured ``web`` directory (see :doc:`web`) to make it available from the web.
