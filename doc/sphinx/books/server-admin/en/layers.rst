Layers
======

A *layer* on a GBD WebSuite project is identified by its ``type``, in addition, layers have the following properties (if not explcitly configured, there will be inherited from the parent layer or from the map):

* ``display`` - display this layer as a single image (``box``) or tiled (``tile``)
* ``view`` - spatial properties of the layer (extent, set of allowed resolutions or scales for this layer)
* ``cache`` and ``grid`` - affect the layer caching (see :doc:`cache`)
* ``clientOptions`` - options for the GBD WebSuite client (see :doc:`client`)
* ``meta`` - layer meta-data (e.g. attribution)
* ``featureFormat`` - formatting and transformation rules for features (see :doc:`features`)


External service layers
-----------------------

These layers display map imagery from remote services. All these types require an url of the service in the configuration.


wms
~~~

You can speicify which layers you want to use. All WMS layers are reprojected, combined and displayed as a single image ::

    {
        "type": "wms",
        "title": "Webatlas.de",
        "sourceLayers": {
            "names": ["dtk250"]
        },
        "url": "http://sg.geodatenzentrum.de/wms_dtk250"
    }

wmts
~~~~

If the service provides multiple layers, you can specify which layer to use ::


    {
        "type": "wmts",
        "title": "NRW geobasis",
        "sourceLayer": "DOP",
        "url": "https://www.wmts.nrw.de/geobasis/wmts_nw_dop"
    }

tile
~~~~

The service url must contain the placeholders ``{x}``, ``{y}`` and ``{z}`` ::

    {
        "type": "tile",
        "title": "Wikimaps",
        "url": "https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png"
    }


QGIS layers
-----------

These layers are rendered using the bundled QGIS server. The mandatory parameter ``path`` tells where the Qgis project is located. Note that the Qgis option "Use layer IDs as WMS names" should be "on" in for all Qgis projects.


qgis
~~~~

QGIS layers display whole qgis projects as a single group in the GWS layer tree. In addition to a layer filter, you can specify whether remote (e.g. WMS) layers should be rendered and/or searched directly, or using the Qgis Server ::


    {
        "type": "qgis",
        "title": "My qgis project",
        "path": "/data/path/to/my-project.qgis",
        "directRender": ["wms"]
    }


qgisflat
~~~~~~~

QGIS/WMS layers display individual layers from a Qgis project as a single flat image ::

    {
        "type": "qgisflat",
        "title": "My qgis project",
        "path": "/data/path/to/my-project.qgis",
        "sourceLayers": [
            "names": ["My First Layer", "My Second Layer"]
        ]
    }


Vector layers
-------------

These layers take geometry data from external sources and display it as vectors using client-side Javascript. You can configure the appearance of the layer using the ``style`` option.

sql
~~~

SQL layers take geometries from an SQL table. You need to specify the database provider and the table to use ::

        {
            "title": "Houses",
            "type": "sql",
            "table": {
                "name": "myschema.mytable",
                "keyColumn": "id",
                "geometryColumn": "geom"
            },
            "loadingStrategy": "bbox",
            "style": {
                "type": "css",
                "text": "stroke:rgb(0,255,0); stroke-width:1px; fill:rgba(0,100,0,0.2)"
            }
        }


Other layers
------------

group
~~~~~

Group layers contain other layers, they don't provide any geodata by themselves. A group can be made "virtual", or ``unfolded``, in which case it's not displayed in the client, while its child layers are ::

    {
        "type": "group",
        "title": "Background",
        "layers": [
            ...
        ]
    }

