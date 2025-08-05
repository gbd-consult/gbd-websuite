"""MapServer support.

This module dynamically creates and renders MapServer maps.

To render a map, create a map object with `new_map`, add layers to it using ``add_`` methods
and invoke ``draw``.

Reference: MapServer documentation (https://mapserver.org/documentation.html)

Example usage::

    import gws.gis.ms as ms

    # create a new map
    map = ms.new_map()

    # add a raster layer from an image file
    map.add_layer(
        ms.LayerOptions(
            type=ms.LayerType.raster,
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

from typing import Optional
import mapscript

import gws
import gws.lib.image


def version() -> str:
    """Returns the MapServer version string."""

    return mapscript.msGetVersion()


class Error(gws.Error):
    pass


class LayerType(gws.Enum):
    """MapServer layer type."""

    point = 'point'
    line = 'line'
    polygon = 'polygon'
    raster = 'raster'


_LAYER_TYPE_TO_MS = {
    LayerType.point: mapscript.MS_LAYER_POINT,
    LayerType.line: mapscript.MS_LAYER_LINE,
    LayerType.polygon: mapscript.MS_LAYER_POLYGON,
    LayerType.raster: mapscript.MS_LAYER_RASTER,
}


class LayerOptions(gws.Data):
    """Options for a mapserver layer."""

    type: LayerType
    """Layer type."""
    path: str
    """Path to the image file."""
    tileIndex: str
    """Path to the tile index SHP file"""
    crs: gws.Crs
    """Layer CRS."""
    connectionType: str
    """Type of connection (e.g., 'postgres')."""
    connectionString: str
    """Connection string for the data source."""
    dataString: str
    """Layer DATA option."""
    style: gws.StyleValues
    """Style for the layer."""
    processing: list[str]
    """Processing options for the layer."""
    transparentColor: str
    """Color to treat as transparent in the layer (OFFSITE)."""
    sldPath: str
    """Path to SLD file for styling the layer."""
    sldName: str
    """Name of an SLD NamedLayer to apply."""


def new_map(config: str = '') -> 'Map':
    """Creates a new Map instance from a Mapfile string."""

    return Map(config)


class Map:
    """MapServer map object wrapper."""

    mapObj: mapscript.mapObj

    def __init__(self, config: str = ''):
        if config:
            tmp = gws.c.EPHEMERAL_DIR + '/mapse_' + gws.u.random_string(16) + '.map'
            gws.u.write_file(tmp, config)
            self.mapObj = mapscript.mapObj(tmp)
        else:
            self.mapObj = mapscript.mapObj()

        # self.mapObj.setConfigOption('MS_ERRORFILE', 'stderr')
        # self.mapObj.debug = mapscript.MS_DEBUGLEVEL_DEVDEBUG

    def copy(self) -> 'Map':
        """Creates a copy of the current map object."""

        c = Map()
        c.mapObj = self.mapObj.clone()
        return c

    def add_layer_from_config(self, config: str) -> mapscript.layerObj:
        """Adds a layer to the map using a configuration string."""

        try:
            lo = mapscript.layerObj()
            lo.updateFromString(config)
            self.mapObj.insertLayer(lo)  # type: ignore
            return lo
        except mapscript.MapServerError as exc:
            raise Error(f'ms: add error:: {exc}') from exc

    def add_layer(self, opts: LayerOptions) -> mapscript.layerObj:
        """Adds a layer to the map."""

        try:
            lo = self._make_layer(opts)
            self.mapObj.insertLayer(lo)  # type: ignore
            return lo
        except mapscript.MapServerError as exc:
            raise Error(f'ms: add error:: {exc}') from exc

    def _make_layer(self, opts: LayerOptions) -> mapscript.layerObj:
        lo = mapscript.layerObj()
        lc = self.mapObj.numlayers
        lo.name = f'_gws_{lc}'
        lo.status = mapscript.MS_ON

        if not opts.crs:
            raise Error('missing layer CRS')
        lo.setProjection(opts.crs.epsg)

        if opts.type:
            lo.type = _LAYER_TYPE_TO_MS[opts.type]
        if opts.path:
            lo.data = opts.path
        if opts.tileIndex:
            lo.tileindex = opts.tileIndex
        if opts.processing:
            for p in opts.processing:
                lo.addProcessing(p)
        if opts.transparentColor:
            co = mapscript.colorObj()
            co.setHex(opts.transparentColor)
            lo.offsite = co
        if opts.connectionType:
            if opts.connectionType == 'postgres':
                lo.setConnectionType(mapscript.MS_POSTGIS, '')
            raise Error(f'unsupported connectionType {opts.connectionType!r}')
        if opts.connectionString:
            lo.connection = opts.connectionString
        if opts.dataString:
            lo.data = opts.dataString
        if opts.sldPath:
            lo.applySLD(gws.u.read_file(opts.sldPath), opts.sldName)

        # @TODO: support style values
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

        try:
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

        except mapscript.MapServerError as exc:
            raise Error(f'ms: draw error: {exc}') from exc

    def to_string(self) -> str:
        """Converts the map object to a configuration string."""

        try:
            return self.mapObj.convertToString()
        except mapscript.MapServerError as exc:
            raise Error(f'ms: convert error: {exc}') from exc
