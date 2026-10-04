"""Raster layers from georeferenced image files.

This plugin provides the ``raster`` layer, which shows georeferenced image
files (e.g. GeoTIFF) from the file system. The images are rendered in
process by MapServer; no external service is involved.

Submodules
----------

- ``provider`` - the image set (`provider.Object`). It lists the image
  files, given as ``paths`` or a glob ``pathPattern``, reads their bounds
  with GDAL and builds a shapefile tile index for MapServer.
- ``layer`` - the ``raster`` image layer. It configures the MapServer layer
  options (``processing`` directives, ``transparentColor``, SLD styling)
  and computes the layer extent from the image bounds, unless an extent is
  configured.
- ``grabber`` - the raster grabber, which renders boxes with MapServer.
  Boxes are requested in the target CRS, MapServer reprojects the images.

All images of a layer must be in the same CRS; images in other CRS, and
images that cannot be opened, are skipped with a configuration warning.
Images without a CRS get the provider ``crs``, or the map CRS.

Example::

    map.layers+ {
        title "Orthophotos"
        type "raster"
        display "tile"
        provider.pathPattern "/data/dop/*.tif"
        provider.crs 25832
        processing [ "BANDS=1,2,3" ]
    }
"""
