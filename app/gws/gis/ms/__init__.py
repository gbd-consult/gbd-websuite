import mapscript

import gws
import gws.lib.image


def version() -> str:
    """Returns the MapServer version string.

    Returns:
        str: The version of MapServer.
    """
    return mapscript.msGetVersion()


class RasterLayerOptions(gws.Data):
    """Represents options for configuring a raster layer."""
    path: str
    tileIndex: str
    bounds: gws.Bounds
    crs: gws.Crs
    processing: list[str]


class VectorLayerOptions(gws.Data):
    """Represents options for configuring a vector layer."""
    geometryType: gws.GeometryType
    connectionType: str
    connectionString: str
    dataString: str
    crs: gws.Crs
    style: gws.StyleValues
    config: str


def new_map() -> 'Map':
    """Creates a new Map instance.

    Returns:
        Map: A new instance of the Map class.
    """
    return Map()


class Map:
    """Represents a MapServer map object."""

    mapObj: mapscript.mapObj

    def __init__(self):
        """Initializes a new Map object with default settings."""
        self.mapObj = mapscript.mapObj()
        # Use PNG by default
        self.mapObj.setOutputFormat(mapscript.outputFormatObj('AGG/PNG'))
        self.mapObj.outputformat.transparent = mapscript.MS_TRUE

    def copy(self) -> 'Map':
        """Creates a copy of the current map object.

        Returns:
            Map: A new instance with the same properties as the current map.
        """
        c = Map()
        c.mapObj = self.mapObj.clone()
        return c

    def add_raster_layer(self, opts: RasterLayerOptions) -> mapscript.layerObj:
        """Adds a raster layer to the map.

        Args:
            opts: Configuration options for the raster layer.

        Returns:
            mapscript.layerObj: The created raster layer object.
        """
        lo = mapscript.layerObj(self.mapObj)

        lc = self.mapObj.numlayers
        lo.name = f'_gws_{lc}'
        lo.type = mapscript.MS_LAYER_RASTER
        lo.status = mapscript.MS_ON

        if opts.path:
            lo.data = opts.path
        if opts.tileIndex:
            lo.tileindex = opts.tileIndex

        if opts.processing:
            for p in opts.processing:
                lo.addProcessing(p)

        lo.setProjection(f'init=epsg:{opts.crs.srid}')

        self.mapObj.insertLayer(lo)

        return lo

    def add_vector_layer(self, opts: VectorLayerOptions) -> mapscript.layerObj:
        """Adds a vector layer to the map.

        Args:
            opts: Configuration options for the vector layer.

        Returns:
            mapscript.layerObj: The created vector layer object.

        Raises:
            gws.Error: If an invalid vector layer type is specified.
        """
        lo = mapscript.layerObj(self.mapObj)

        lc = self.mapObj.numlayers
        lo.name = f'_gws_{lc}'
        lo.status = mapscript.MS_ON

        if opts.geometryType:
            lo.type = _GEOM_TO_MS_LAYER[opts.geometryType]

        if opts.connectionType:
            if opts.connectionType == 'postgres':
                lo.setConnectionType(mapscript.MS_POSTGIS, '')
            raise gws.Error(f'Invalid connectionType {opts.connectionType!r}')

        if opts.connectionString:
            lo.connection = opts.connectionString

        if opts.dataString:
            lo.data = opts.dataString

        if opts.crs:
            lo.setProjection(f'init=epsg:{opts.crs.srid}')

        if opts.config:
            lo.updateFromString(opts.config)

        self.mapObj.insertLayer(lo)
        return lo

    def draw(self, bounds: gws.Bounds, size: gws.Size) -> gws.lib.image.Image:
        """Renders the map within the given bounds and size.

        Args:
            bounds: The spatial extent to render.
            size: The output image size.

        Returns:
            gws.lib.image.Image: The rendered map image.
        """
        gws.debug.time_start(f'mapserver.draw {bounds=} {size=}')
        self.mapObj.setExtent(*bounds.extent)
        self.mapObj.setSize(*size)
        self.mapObj.setProjection(bounds.crs.epsg)
        res = self.mapObj.draw()
        img = gws.lib.image.from_bytes(res.getBytes())
        gws.debug.time_end()
        return img

    def to_string(self) -> str:
        """Converts the map object to a string representation.

        Returns:
            str: The string representation of the map object.
        """
        return self.mapObj.convertToString()


_GEOM_TO_MS_LAYER = {
    gws.GeometryType.point: mapscript.MS_LAYER_POINT,
    gws.GeometryType.linestring: mapscript.MS_LAYER_LINE,
    gws.GeometryType.polygon: mapscript.MS_LAYER_POLYGON,
}
