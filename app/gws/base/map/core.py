from typing import Optional

import gws
import gws.base.layer
import gws.gis.crs
import gws.gis.bounds
import gws.gis.extent
import gws.gis.zoom
import gws.lib.uom as units

gws.ext.new.map('default')


class Config(gws.Config):
    """Map configuration"""

    center: Optional[gws.Point]
    """map center"""
    coordinatePrecision: Optional[int]
    """precision for coordinates"""
    crs: Optional[gws.CrsName] = 'EPSG:3857'
    """crs for this map"""
    extent: Optional[gws.Extent]
    """map extent"""
    extentBuffer: Optional[int]
    """extent buffer"""
    layers: list[gws.ext.config.layer]
    """collection of layers for this map"""
    title: str = ''
    """map title"""
    wrapX: bool = False
    """wrap the world horizontally"""
    zoom: Optional[gws.gis.zoom.Config]
    """map scales and resolutions"""


class Props(gws.Data):
    crs: str
    crsDef: Optional[str]
    coordinatePrecision: int
    extent: gws.Extent
    center: gws.Point
    initResolution: float
    rootLayer: gws.base.layer.Props
    resolutions: list[float]
    title: str = ''
    wrapX: bool


class _RootLayer(gws.base.layer.group.Object):
    parent: 'Object'


class Object(gws.Map):
    wrapX: bool

    def configure(self):
        self.title = self.cfg('title') or self.cfg('_defaultTitle') or ''

        p = self.cfg('crs')
        crs = gws.gis.crs.require(p) if p else gws.gis.crs.WEBMERCATOR

        p = self.cfg('extent')
        if p:
            self.bounds = gws.Bounds(
                crs=crs,
                extent=gws.gis.extent.buffer(gws.gis.extent.from_list(p), self.cfg('extentBuffer') or 0)
            )
        else:
            self.bounds = gws.Bounds(crs=crs, extent=crs.extent)

        self.center = self.cfg('center') or gws.gis.extent.center(self.bounds.extent)
        self.wgsExtent = gws.gis.extent.transform_to_wgs(self.bounds.extent, self.bounds.crs)
        self.wrapX = self.cfg('wrapX', default=False)

        p = self.cfg('zoom')
        if p:
            self.resolutions = gws.gis.zoom.resolutions_from_config(p)
            self.initResolution = gws.gis.zoom.init_resolution(p, self.resolutions)
        else:
            self.resolutions = gws.gis.zoom.OSM_RESOLUTIONS
            self.initResolution = self.resolutions[len(self.resolutions) >> 1]

        p = self.cfg('coordinatePrecision')
        if p:
            self.coordinatePrecision = p
        else:
            self.coordinatePrecision = (2 if self.bounds.crs.uom == gws.Uom.m else 7)

        self.rootLayer = self.create_child(
            gws.ext.object.layer,
            type='group',
            title=self.title,
            layers=self.cfg('layers'),
            _parentBounds=self.bounds,
            _parentResolutions=self.resolutions,
        )

        if not self.rootLayer:
            raise gws.Error(f'missing or invalid root layer in {self!r}')

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
            wrapX=self.wrapX,
        )
