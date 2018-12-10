Projects
========

A *project* in GWS consists of a map, print templates and additional options. Besides that, you can override some application options for each project individually.

Project locations
-----------------

There are three ways to add projects to your GWS installation (you can also combine them):

TABLE
``projects`` ~ configure projects directly in the main configuration
``projectPaths`` ~ include specific project configurations
``projectDirs`` ~ include all projects from directories
/TABLE

When using ``projectDirs``, the server searches the specified directories recursively and adds all files that end with ``config.py`` or ``config.json`` or ``config.yaml``.

Each project configuration file can contain a configuration for a single project, or an array of such configurations.

Project configurations
----------------------

A project configuration should contain at least the ``title`` and a ``Map`` configuration. Some options, like ``access``, ``assets`` and ``client`` override respective application and website options.  ``printer`` describes project print templates (see :doc:`print`).


Maps
----

A *map* is basically a collection of *layers* (see :doc:`layers`). A map configuration can also contain Cache, Grid and View options that serve as fallback for layers that don't define them explicitly. The ``crs`` option must be valid EPSG CRS reference string. All layers in the project will be displayed in the CRS. Sources with different projections will be dynamically reprojected.

Multi-projects
--------------

A project configuration can also serve as a template for multiple projects. To set up a template, set ``multi`` to ``true`` and provide a regular expression in ``multiMatch``. The server enumerates all files on the server that match that expression, and creates a project config for each file by replacing regex placeholders ``{$n}`` in other options' values. For example, this template will enumerate all QGIS projects in ``/data/qgis-maps`` and create a project with a qgis layer for each map found ::


        "multi": True,
        "multiMatch": "/data/qgis-maps/(.+?).qgs$",
        "title": "Project {$1}",
        "map": {
            "layers": [
                {
                    "type": "qgis",
                    "title": "{$1}",
                    "source": {
                        "type": "qgis",
                        "map": "{$0}"
                    },
                },
            ]
        }









