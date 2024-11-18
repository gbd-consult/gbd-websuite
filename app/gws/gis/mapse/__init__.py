import mapscript

import gws
import gws.lib.image

_GEOM_TO_MS_LAYER = {
    gws.GeometryType.point: mapscript.MS_LAYER_POINT,
    gws.GeometryType.linestring: mapscript.MS_LAYER_LINE,
    gws.GeometryType.polygon: mapscript.MS_LAYER_POLYGON,
}

class RasterLayerOptions(gws.Data):
    path: str
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
    mo: mapscript.mapObj
    layerCnt: int

    def __init__(self):
        self.layerCnt = 0

    def init_from_bounds(self, bounds: gws.Bounds) -> 'Map':
        self.mo = mapscript.mapObj()

        self.mo.setProjection(f'init=epsg:{bounds.crs.srid}')
        self.mo.setOutputFormat(mapscript.outputFormatObj('AGG/PNG'))
        self.mo.outputformat.transparent = mapscript.MS_TRUE

        return self

    def add_raster_layer(self, opts: RasterLayerOptions) -> 'Map':
        lo = mapscript.layerObj(self.mo)

        self.layerCnt += 1
        lo.name = f'_{self.layerCnt}'
        lo.type = mapscript.MS_LAYER_RASTER
        lo.status = mapscript.MS_ON
        lo.data = opts.path

        lo.setProjection(f'init=epsg:{opts.crs.srid}')
        # lo.setExtent(*opts.bounds.extent)
        lo.addProcessing('NODATA=0')

        self.mo.insertLayer(lo)

        return self

    def add_vector_layer(self, opts: VectorLayerOptions) -> 'Map':
        lo = mapscript.layerObj(self.mo)

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

        self.mo.insertLayer(lo)

        return self




    def draw(self, bounds: gws.Bounds, size: gws.Size) -> gws.lib.image.Image:
        self.mo.setExtent(*bounds.extent)
        self.mo.setSize(*size)
        im = self.mo.draw()
        return gws.lib.image.from_bytes(im.getBytes())


def map_from_bounds(bounds: gws.Bounds) -> Map:
    m = Map()
    return m.init_from_bounds(bounds)
