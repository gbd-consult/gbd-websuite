"""Tile layers from XYZ tile services.

This plugin provides the ``tile`` layer, which shows tiles from a REST tile
service whose URL contains the placeholders ``{x}``, ``{y}`` and ``{z}``.

Submodules
----------

- ``provider`` - the tile service (`provider.Object`). It holds the URL
  template, the finest source level and the tile grid of the source
  (web mercator by default), and fetches single tiles.
- ``layer`` - the ``tile`` image layer. Its extent is the source grid
  extent, clipped to the area of use of the grid CRS.
- ``grabber`` - the tile grabber. It fetches source tiles through the
  provider and mosaics and warps them onto the requested tiles and boxes.
  The source tile matrix set is derived from the provider grid, with one
  matrix per level up to the provider's ``maxLevel``.

The layer supports the usual display modes. With ``display "tile"`` (the
default) or ``"box"``, images are served by WebSuite through the grabber and
can be cached. With ``display "client"``, the client loads the tiles
directly from the source URL, as an ``xyz`` layer.

The source grid is assumed to have its origin in the north-west corner
(XYZ order); TMS sources with a south-west origin are not supported.

Example::

    map.layers+ {
        title "OpenStreetMap"
        type "tile"
        provider.url "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        display "tile"
        withCache true
    }
"""
