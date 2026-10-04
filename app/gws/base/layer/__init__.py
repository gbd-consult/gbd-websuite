"""Base classes and utilities for map layers.

A layer is a node in the layer tree of a map. It knows its extent, the
resolutions it is visible at, its legend, metadata, templates, models and
search providers, and it can render itself as a box image, as tiles or as an
SVG fragment. Concrete layer types (QGIS, WMS, WFS, tile services, database
tables and so on) live in plugins and subclass the base classes here.

Submodules:

- ``core``: the base ``Object`` with the common ``Config`` and ``Props``. It
  defines the configuration protocol ``configure_layer``, which calls a fixed
  sequence of ``configure_*`` steps (provider, sources, group, models, extent,
  bounds, zoom bounds, resolutions, legend, metadata, templates, search, OWS).
  Each step returns ``True`` when it has set its value, so a subclass can
  override a step, call ``super()`` first and fill in a default only if the
  base step did nothing.
- ``group``: the ``group`` layer type, which holds child layers and derives
  its extent, resolutions, legend and render capabilities from them. Unless
  configured explicitly, the extent, zoom bounds and resolutions are the union
  of those of the children, and the legend is a ``combined`` legend of the
  children that have one. The group can render boxes, tiles or SVG if any
  child can, and is searchable if any child is.
- ``image``: base class for raster layers. It creates one grabber
  (see ``gws.base.grabber``) per CRS supported by the application and renders
  boxes and tiles through the grabber for the target CRS. Subclasses provide
  ``create_grabber`` and ``create_cache_name``. Rotated boxes are rendered as
  a larger square, then rotated and cropped.
- ``vector``: base class for vector layers. It finds features through the
  first model the user can read and renders them as SVG.
- ``ows``: the binding of a layer to OWS services (layer and feature names,
  XML namespace, allowed services, OWS models). Names may carry an XML
  namespace prefix (``prefix:name``), which then sets the XML namespace.
  Layer and feature names default to the layer title converted to a UID, the
  geometry name defaults to an empty string.
- ``tree``: builds child layer configurations from a hierarchy of source
  layers, for layer types that mirror an external layer tree (QGIS projects,
  WMS and WFS services). Its ``Config`` adds the ``rootLayers``,
  ``excludeLayers``, ``flattenLayers`` and ``autoLayers`` options.

``Layer.render`` dispatches to ``render_box``, ``render_tile`` or
``render_svg`` by the input type. ``Layer.url_path_for`` builds the URLs of
the ``mapGetBox``, ``mapGetTile``, ``mapGetLegend`` and ``mapGetFeatures``
commands of the map action. ``Layer.render_legend`` without arguments caches
the legend for the server lifetime.

Child layers receive the parent's WGS extent, resolutions and the map CRS
through the internal config keys ``_parentWgsExtent``, ``_parentResolutions``
and ``_mapCrs``; a layer's own extent is clipped to the parent extent.

Example::

    map.layers+ {
        type "group"
        title "Base maps"
        layers+ {
            type "tile"
            title "OpenStreetMap"
            provider.url "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        }
    }

A minimal raster layer type in a plugin::

    @gws.ext.object.layer('mytiles')
    class Object(gws.base.layer.image.Object):
        def configure(self):
            self.configure_layer()

        def create_cache_name(self, cache):
            return gws.u.sha256(self.uid)[: gws.base.layer.image.CACHE_NAME_LENGTH]

        def create_grabber(self, opts):
            return MyGrabber(opts)
"""

from .core import (
    Object,
    Config,
    Props,
)

from . import group, tree, image, vector
