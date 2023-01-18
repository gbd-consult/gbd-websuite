import gws
import gws.base.layer
import gws.gis.crs
import gws.gis.bounds
import gws.gis.extent
import gws.gis.zoom
import gws.lib.uom as units
import gws.types as t


@gws.ext.config.map('default')
class Config(gws.Config):
    """Map configuration"""

    center: t.Optional[gws.Point]
    """map center"""
    coordinatePrecision: t.Optional[int]
    """precision for coordinates"""
    crs: t.Optional[gws.CrsName] = 'EPSG:3857'
    """crs for this map"""
    extent: t.Optional[gws.Extent]
    """map extent"""
    extentBuffer: t.Optional[int]
    """extent buffer"""
    layers: t.List[gws.ext.config.layer]
    """collection of layers for this map"""
    title: str = ''
    """map title"""
    zoom: t.Optional[gws.gis.zoom.Config]
    """map scales and resolutions"""


@gws.ext.props.map('default')
class Props(gws.Data):
    crs: str
    crsDef: t.Optional[str]
    coordinatePrecision: int
    extent: gws.Extent
    center: gws.Point
    initResolution: float
    rootLayer: gws.base.layer.Props
    resolutions: t.List[float]
    title: str = ''


@gws.ext.object.map('default')
class Object(gws.Node, gws.IMap):

    def configure(self):
        self.title = self.var('title') or ''

        p = self.var('crs')
        crs = gws.gis.crs.require(p) if p else gws.gis.crs.WEBMERCATOR

        p = self.var('extent')
        if p:
            self.bounds = gws.Bounds(crs=crs, extent=gws.gis.extent.from_list(p))
        else:
            self.bounds = gws.Bounds(crs=crs, extent=crs.extent)

        self.center = self.var('center') or gws.gis.extent.center(self.bounds.extent)
        self.wgsExtent = gws.gis.extent.transform_to_wgs(self.bounds.extent, self.bounds.crs)

        p = self.var('zoom')
        if p:
            self.resolutions = gws.gis.zoom.resolutions_from_config(p, gws.gis.zoom.OSM_RESOLUTIONS)
            self.initResolution = gws.gis.zoom.init_resolution(p, self.resolutions)
        else:
            self.resolutions = gws.gis.zoom.OSM_RESOLUTIONS
            self.initResolution = self.resolutions[len(self.resolutions) >> 1]

        p = self.var('coordinatePrecision')
        if p:
            self.coordinatePrecision = p
        else:
            self.coordinatePrecision = (2 if self.bounds.crs.uom == gws.Uom.m else 7)

        self.rootLayer = self.create_child(gws.ext.object.layer, gws.Config(
            type='group',
            layers=self.var('layers'),
            _parentBounds=self.bounds,
            _parentResolutions=self.resolutions,
        ))

    def props(self, user):
        return gws.Data(
            crs=self.bounds.crs.epsg,
            crsDef=self.bounds.crs.proj4text,
            coordinatePrecision=self.coordinatePrecision,
            extent=self.bounds.extent,
            center=self.center,
            initResolution=self.initResolution,
            rootLayer=self.rootLayer,
            resolutions=sorted(self.resolutions, reverse=True),
            title=self.title,
        )

# def _configure_extent(layer: gws.ILayer, crs: gws.ICrs, default_extent):
#     # we have an explicit extent provided in the config
#
#     p = layer.var('extent')
#     if p:
#         layer.extent = gws.gis.extent.from_list(p)
#         if not layer.extent:
#             raise gws.Error(f'{layer.uid!r}: invalid extent {p!r}')
#
#         # configure sublayers using config_ext as a default
#         for la in layer.layers:
#             _configure_extent(la, crs, layer.extent)
#
#         return layer.extent
#
#     if layer.layers:
#         # no config extent, configure sublayers using the current default extent
#         # set obj.extent to the sum of sublayers' extents
#
#         ext_list = []
#         for la in layer.layers:
#             layer_ext = _configure_extent(la, crs, default_extent)
#             if layer_ext:
#                 ext_list.append(layer_ext)
#
#         layer.extent = gws.gis.extent.merge(ext_list) if ext_list else default_extent
#         return layer.extent
#
#     # obj is a leaf layer and has no configured extent
#     # check if it has an own extent (from its source)
#
#     own_bounds = layer.own_bounds()
#     if own_bounds:
#         own_ext = own_bounds.extent
#         buf = layer.var('extentBuffer')
#         if buf:
#             own_ext = gws.gis.extent.buffer(own_ext, buf)
#         layer.extent = gws.gis.extent.transform(own_ext, own_bounds.crs, crs)
#         return layer.extent
#
#     # obj is a leaf layer and has neither configured nor own extent
#     # try using the default extent
#
#     if default_extent:
#         layer.extent = default_extent
#         return layer.extent
#
#     # no extent can be computed, it will be set to the map extent later on
#     return None
#
#
# def _set_default_extentent(layer: gws.ILayer, extent):
#     layer.extent = layer.extent or extent
#     for la in layer.layers:
#         _set_default_extentent(la, extent)
