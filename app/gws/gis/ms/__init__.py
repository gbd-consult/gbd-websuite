"""MapServer support.

This module dynamically creates and renders MapServer maps. 

To render a map, create a map object with `new_map`, add layers to it using ``add_`` methods 
and invoke ``draw``.

Reference: MapServer documentation (https://mapserver.org/documentation.html)

Example usage::

    import gws.gis.ms as ms

    # creare a new map
    map = ms.new_map()

    # add a raster layer from an image file
    map.add_raster_layer(
        ms.RasterLayerOptions(
            path="/path/to/image.tif",
        )
    )

    # add a layer using a configuration string
    map.add_layer('''
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
            extent=[738040 ,6653804, 765743, 6683686],
            crs=gws.lib.crs.WEBMERCATOR,
        ),
        size=(800, 600),
    )

    # save the image to a file
    img.to_path('/path/to/output.png')


"""

from typing import Optional
import mapscript

import gws
import gws.lib.image


def version() -> str:
    """Returns the MapServer version string."""

    return mapscript.msGetVersion()


class Error(gws.Error):
    pass


class RasterLayerOptions(gws.Data):
    """Options for a raster layer."""

    path: str
    """Path to the image file."""
    tileIndex: Optional[str]
    """Path to the tile index SHP file"""
    crs: gws.Crs
    """Layer CRS."""
    processing: list[str]
    """Processing options"""


class VectorLayerOptions(gws.Data):
    """Options for a vector layer."""

    geometryType: gws.GeometryType
    """Layer geometry type."""
    connectionType: str
    """Type of connection (e.g., 'postgres')."""
    connectionString: str
    """Connection string for the data source."""
    dataString: str
    """Layer DATA option."""
    crs: gws.Crs
    """Layer CRS."""
    style: gws.StyleValues
    """Style for the layer."""


def new_map() -> 'Map':
    """Creates a new Map instance.

    Returns:
        Map: A new instance of the Map class.
    """
    return Map()


class Map:
    """MapServer map object."""

    mapObj: mapscript.mapObj

    def __init__(self):
        self.mapObj = mapscript.mapObj()

    def copy(self) -> 'Map':
        """Creates a copy of the current map object."""

        c = Map()
        c.mapObj = self.mapObj.clone()
        return c

    def add_layer(self, config: str) -> mapscript.layerObj:
        """Adds a layer to the map using a configuration string.

        Args:
            config: Raw MapServer configuration string.

        Returns:
            The created layer object.
        """

        lo = mapscript.layerObj(self.mapObj)
        lo.updateFromString(config)
        self.mapObj.insertLayer(lo)  # type: ignore
        return lo

    def add_raster_layer(self, opts: RasterLayerOptions) -> mapscript.layerObj:
        """Adds a raster layer to the map.

        Args:
            opts: Configuration options for the raster layer.

        Returns:
            The created raster layer object.
        """

        lo = self._insert_layer(opts)
        lo.type = mapscript.MS_LAYER_RASTER

        if opts.path:
            lo.data = opts.path
        if opts.tileIndex:
            lo.tileindex = opts.tileIndex
        if opts.processing:
            for p in opts.processing:
                lo.addProcessing(p)

        return lo

    def add_vector_layer(self, opts: VectorLayerOptions) -> mapscript.layerObj:
        """Adds a vector layer to the map.

        Args:
            opts: Configuration options for the vector layer.

        Returns:
            The created vector layer object.
        """
        lo = self._insert_layer(opts)

        if opts.geometryType:
            if opts.geometryType not in _GEOM_TO_MS_LAYER:
                raise Error(f'unsupported geometryType {opts.geometryType!r}')
            lo.type = _GEOM_TO_MS_LAYER[opts.geometryType]

        if opts.connectionType:
            if opts.connectionType == 'postgres':
                lo.setConnectionType(mapscript.MS_POSTGIS, '')
            raise Error(f'unsupported connectionType {opts.connectionType!r}')

        if opts.connectionString:
            lo.connection = opts.connectionString

        if opts.dataString:
            lo.data = opts.dataString

        # @TODO: support style values

        return lo

    def _insert_layer(self, opts: gws.Data):
        lo = mapscript.layerObj(self.mapObj)

        lc = self.mapObj.numlayers
        lo.name = f'_gws_{lc}'
        lo.status = mapscript.MS_ON

        if not opts.crs:
            raise Error('missing crs')
        lo.setProjection(opts.crs.epsg)

        self.mapObj.insertLayer(lo)  # type: ignore

        return lo

    def draw(self, bounds: gws.Bounds, size: gws.Size) -> gws.Image:
        """Renders the map within the given bounds and size.

        Args:
            bounds: The spatial extent to render.
            size: The output image size.

        Returns:
            The rendered map image.
        """

        # @TODO: options for image format, transparency, etc.

        gws.debug.time_start(f'mapserver.draw {bounds=} {size=}')

        self.mapObj.setOutputFormat(mapscript.outputFormatObj('AGG/PNG'))
        self.mapObj.outputformat.transparent = mapscript.MS_TRUE

        self.mapObj.setExtent(*bounds.extent)
        self.mapObj.setSize(*size)
        self.mapObj.setProjection(bounds.crs.epsg)

        res = self.mapObj.draw()
        img = gws.lib.image.from_bytes(res.getBytes())

        gws.debug.time_end()

        return img

    def to_string(self) -> str:
        """Converts the map object to a configuration string.

        Returns:
            The string representation of the map object.
        """
        return self.mapObj.convertToString()


_GEOM_TO_MS_LAYER = {
    gws.GeometryType.point: mapscript.MS_LAYER_POINT,
    gws.GeometryType.linestring: mapscript.MS_LAYER_LINE,
    gws.GeometryType.polygon: mapscript.MS_LAYER_POLYGON,
}
