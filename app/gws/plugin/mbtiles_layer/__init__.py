"""MBTiles layer.

Raster layer from an MBTiles file. The file is opened as a GDAL raster and
rendered with MapServer by a box grabber; MapServer reprojects the raster to
the requested CRS. The CRS of the layer source and, unless configured, the
extent of the layer are taken from the file. ``processing`` passes MapServer
PROCESSING directives, ``transparentColor`` sets a color that is rendered as
transparent.

Submodules
----------

- ``layer`` - the ``mbtiles`` layer.
- ``provider`` - holds the path of the MBTiles file.
- ``grabber`` - the box grabber that renders the file with MapServer.

Example::

    map.layers+ {
        title "TK10"
        type "mbtiles"
        provider.path "/data/tk10.mbtiles"
        display "tile"
        transparentColor "#ffffff"
        processing ["SCALE=0,240"]
    }
"""
