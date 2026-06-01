"""MapServer support.

This module dynamically creates and renders MapServer maps.

To render a map, create a map object with `new_map`, add layers to it using ``add_`` methods
and invoke ``draw``.

Reference: MapServer documentation (https://mapserver.org/documentation.html)

Example usage::

    import gws
    import gws.lib.mapserver as ms

    # create a new map
    map = ms.new_map()

    # add a raster layer from an image file
    map.add_layer(
        gws.MapServerLayerOptions(
            type=gws.MapServerLayerType.raster,
            path='/path/to/image.tif',
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
from . import util
