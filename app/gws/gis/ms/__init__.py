import mapscript

import gws
import gws.lib.image


def version():
    return mapscript.msGetVersion()


class RasterLayerOptions(gws.Data):
    path: str
    tileIndex: str
    bounds: gws.Bounds
    crs: gws.Crs


class VectorLayerOptions(gws.Data):
    type: str
    geometryType: gws.GeometryType
    connectionString: str
    dataString: str
    crs: gws.Crs
    style: gws.StyleValues


def new_map():
    return Map()


class Map:
    mapObj: mapscript.mapObj

    def __init__(self):
        self.mapObj = mapscript.mapObj()
        # use PNG by default
        self.mapObj.setOutputFormat(mapscript.outputFormatObj('AGG/PNG'))
        self.mapObj.outputformat.transparent = mapscript.MS_TRUE



    def copy(self):
        c = Map()
        c.mapObj = self.mapObj.clone()
        return c

    def add_raster_layer(self, opts: RasterLayerOptions) -> mapscript.layerObj:
        lo = mapscript.layerObj(self.mapObj)

        lc = self.mapObj.numlayers
        lo.name = f'_gws_{lc}'
        lo.type = mapscript.MS_LAYER_RASTER
        lo.status = mapscript.MS_ON

        if opts.path:
            lo.data = opts.path
        if opts.tileIndex:
            lo.tileindex = opts.tileIndex

        lo.setProjection(f'init=epsg:{opts.crs.srid}')

        self.mapObj.insertLayer(lo)

        return lo

    def add_vector_layer(self, opts: VectorLayerOptions) -> mapscript.layerObj:
        lo = mapscript.layerObj(self.mapObj)

        lc = self.mapObj.numlayers
        lo.name = f'_gws_{lc}'

        lo.type = _GEOM_TO_MS_LAYER[opts.geometryType]
        lo.status = mapscript.MS_ON

        if opts.type == 'postgres':
            lo.setConnectionType(mapscript.MS_POSTGIS, '')
        else:
            # @TODO
            raise gws.Error(f'invalid vector type {opts.type!r}')

        lo.connection = opts.connectionString
        lo.data = opts.dataString

        lo.setProjection(f'init=epsg:{opts.crs.srid}')

        self.mapObj.insertLayer(lo)

        return lo

    def draw(self, bounds: gws.Bounds, size: gws.Size) -> gws.lib.image.Image:
        self.mapObj.setExtent(*bounds.extent)
        self.mapObj.setSize(*size)
        self.mapObj.setProjection(bounds.crs.epsg)
        im = self.mapObj.draw()
        return gws.lib.image.from_bytes(im.getBytes())

    def to_string(self):
        return self.mapObj.convertToString()


_GEOM_TO_MS_LAYER = {
    gws.GeometryType.point: mapscript.MS_LAYER_POINT,
    gws.GeometryType.linestring: mapscript.MS_LAYER_LINE,
    gws.GeometryType.polygon: mapscript.MS_LAYER_POLYGON,
}
