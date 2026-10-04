"""QGIS support.

This plugin uses QGIS projects as data sources for layers, search, models,
legends and print templates. Projects are stored in files (``.qgs`` or
zipped ``.qgz``) or in a Postgres table (``qgis_projects``, the QGIS
"store project in PostgreSQL" format). Projects are parsed by reading the
project XML directly; no QGIS APIs are used. Rendering, feature info,
legends and printing are done by sending requests to a QGIS Server instance,
whose address comes from the ``server.qgis`` application settings.

Submodules
----------

- ``project`` - loading and storing QGIS projects (`project.Object`,
  `project.Store`), in files or in a Postgres database.
- ``caps`` - the project XML parser. It produces a `caps.Caps` object with
  the project metadata, CRS, extents, source layers, print layouts,
  visibility presets and custom properties, and also parses layer data
  source strings (``parse_datasource``).
- ``provider`` - the service provider (`provider.Object`). It loads the
  project, computes the project bounds, talks to QGIS Server (GetMap,
  GetFeatureInfo and other requests) and creates the configuration of leaf
  layers for the ``qgis`` tree layer.
- ``layer`` - the ``qgis`` layer, a group that shows the project as a tree
  of layers.
- ``flatlayer`` - the ``qgisflat`` layer, which renders selected project
  layers as a single image. By default, it gets a ``qgis`` model and a
  ``qgis`` finder for its queryable source layers and a ``qgis`` legend,
  whose options are merged with the provider's ``defaultLegendOptions``.
- ``grabber`` - the raster grabber for ``qgisflat`` layers and composite
  ``qgis`` layers; it requests boxes from QGIS Server with GetMap. Boxes are
  requested in the target CRS, QGIS Server reprojects as needed.
- ``finder`` - the ``qgis`` finder, which searches project layers by point
  with GetFeatureInfo.
- ``model`` - the ``qgis`` model for features found with the finder.
- ``legend`` - the ``qgis`` legend, rendered with GetLegendGraphic. Rendered
  images are cached for ``cacheMaxAge``.
- ``template`` - the ``qgis`` print template, based on a print layout of
  the project.
- ``cli`` - the ``gws qgis caps`` and ``gws qgis copy`` commands.

Design
------

Layers, finders, models, legends and templates refer to a project through
their ``provider`` configuration. A provider created from the same
configuration is shared between objects. When the provider has
``withWatch`` enabled, it checks the project periodically and reloads the
application when the project changes.

The ``qgis`` layer creates a child layer for each source layer of the
project, keeping the group structure. By default, a child is a
``qgisflat`` layer that renders through QGIS Server. With the provider
options ``directRender`` and ``directSearch``, children based on WMS, WMTS
or XYZ sources are rendered by WebSuite directly (``wmsflat``, ``wmts`` or
``tile`` layers), and WMS, WFS and Postgres sources get their own finders
(and, for Postgres, models) instead of going through QGIS Server. With
``compositeRender``, the ``qgis`` layer renders all visible ``qgisflat``
children as one image with one GetMap request. It then creates grabbers like
an image layer, and the client gets a ``compositeBox`` or ``compositeTile``
layer whose children are ``compositeLeaf`` layers; the client sends the
visible children in ``compositeLayerUids``.

Print templates work as follows. The project is reloaded on each render.
The map is rendered by WebSuite as a PDF.
Label and HTML items of the QGIS layout are treated as WebSuite ``html``
templates, so they can use placeholders such as ``@legend``; if any of them
changes, a temporary copy of the project with the rendered HTML is created.
The layout is then printed by QGIS Server with GetPrint, and the QGIS PDF is
placed over the map PDF, so that grids and other decorations are drawn
above the map. For this to work, the page and the map item of the layout
must be transparent, and since the project is copied, it must use absolute
paths to its assets. Integer map positions and sizes in the layout give the
best alignment.

Project and layer extents
-------------------------

QGIS does not provide complete project and layer extents with respect to
symbology. Only data-based extents are known at parse time. Extents are
computed with the following logic:

- if a project provides an explicit WMS extent (Project Properties ->
  QGIS Server -> WMS), this extent is used as the project render extent
  (``bounds``)
- otherwise, if ``useCanvasExtent`` is true, the canvas extent is used
- otherwise, the project render extent is the union of the layers' data
  extents plus the configured ``extentBuffer``
- if the render extent is empty, the CRS extent is taken
- for layers, the data extent is either an explicit extent (Layer
  Properties -> Metadata -> Extent) or an implicit data extent
- the layer data extent is used as a "zoom" extent, but when rendering a
  layer, the project extent is used

Example::

    map.layers+ {
        title "City"
        type "qgis"
        provider.path "/data/city.qgs"
        provider.directSearch [ "wms" "wfs" "postgres" ]
    }

    map.layers+ {
        title "Districts"
        type "qgisflat"
        provider.path "/data/city.qgs"
        sourceLayers.names [ "districts" ]
    }

    printers+ {
        template {
            type "qgis"
            provider.path "/data/print.qgs"
            index 0
        }
    }
"""

from . import provider, project, caps
