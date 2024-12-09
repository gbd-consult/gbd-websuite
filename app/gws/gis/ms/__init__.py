import mapscript

import gws
import gws.lib.image


def version():
    return mapscript.msGetVersion()


def map_from_bounds(bounds: gws.Bounds) -> 'Map':
    m = Map()
    return m.init_from_bounds(bounds)


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


class Map:
    mapObj: mapscript.mapObj
    layerCnt: int

    def __init__(self):
        self.layerCnt = 0

    def clone(self):
        c = Map()
        c.mapObj = self.mapObj.clone()
        c.layerCnt = self.layerCnt
        return c

    def init_from_bounds(self, bounds: gws.Bounds) -> 'Map':
        self.mapObj = mapscript.mapObj()

        self.mapObj.setProjection(f'init=epsg:{bounds.crs.srid}')

        # use PNG by default
        self.mapObj.setOutputFormat(mapscript.outputFormatObj('AGG/PNG'))
        self.mapObj.outputformat.transparent = mapscript.MS_TRUE


        return self

    def add_raster_layer(self, opts: RasterLayerOptions) -> mapscript.layerObj:
        lo = mapscript.layerObj(self.mapObj)

        self.layerCnt += 1
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

        self.layerCnt += 1
        lo.name = f'_{self.layerCnt}'

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
        im = self.mapObj.draw()
        return gws.lib.image.from_bytes(im.getBytes())


_GEOM_TO_MS_LAYER = {
    gws.GeometryType.point: mapscript.MS_LAYER_POINT,
    gws.GeometryType.linestring: mapscript.MS_LAYER_LINE,
    gws.GeometryType.polygon: mapscript.MS_LAYER_POLYGON,
}
