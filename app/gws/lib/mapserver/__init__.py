"""MapServer support.

Creates and renders MapServer maps dynamically, through the ``mapscript`` Python bindings.

To render a map, create a map object with ``new_map``, optionally from a mapfile string,
add layers to it with ``Map.add_layer`` (from ``gws.MapServerLayerOptions``) or
``Map.add_layer_from_config`` (from a mapfile ``LAYER`` block), and call ``Map.draw``,
which renders a transparent PNG for the given bounds and size and returns it as a
``gws.Image``. ``Map.to_string`` returns the resulting mapfile, which helps with debugging.

Layer options support raster files and tile indexes, PostGIS connections, SLD styling
and a part of the GWS style values (geometry and label styles, markers and icons).

Submodules:

- ``core``: the ``Map`` wrapper and ``new_map``.
- ``live_config``: a standalone development tool, not used by the application. It runs an
  HTTP server with a page to edit a mapfile and render it live (see its ``README.md``).

Reference:
    - https://mapserver.org/documentation.html

Example::

    import gws
    import gws.lib.mapserver as ms

    # create a new map
    map = ms.new_map()

    # add a raster layer from an image file
    map.add_layer(
        gws.MapServerLayerOptions(
            type=gws.MapServerLayerType.raster,
            path='/path/to/image.tif',
            crs=gws.lib.crs.WEBMERCATOR,
        )
    )

    # add a layer using a configuration string
    map.add_layer_from_config('''
        LAYER
            TYPE LINE
            STATUS ON
            FEATURE
                POINTS
                    751539 6669003
                    751539 6672326
                    755559 6672326
                END
            END
            CLASS
                STYLE
                    COLOR 0 255 0
                    WIDTH 5
                END
            END
        END
    ''')

    # draw the map into an Image object
    img = map.draw(
        bounds=gws.Bounds(
            extent=[738040, 6653804, 765743, 6683686],
            crs=gws.lib.crs.WEBMERCATOR,
        ),
        size=(800, 600),
    )

    # save the image to a file
    img.to_path('/path/to/output.png')
"""

from .core import version, Error, new_map, Map
