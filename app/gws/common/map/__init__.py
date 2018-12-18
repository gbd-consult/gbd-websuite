import gws
import gws.types as t

import gws.gis.cache
import gws.gis.proj
import gws.gis.zoom
import gws.gis.layer


class Config(t.Config):
    """map configuration"""

    cache: t.CacheConfig = {}  #: default cache
    center: t.Optional[t.Point]  #: map center
    crs: t.Optional[t.crsref] = 'EPSG:3857'  #: crs for this map
    extent: t.Extent  #: map extent
    grid: t.GridConfig = {}  #: default grid configuration
    layers: t.List[t.ext.gis.layer.Config]  #: collection of layers for this map
    title: t.Optional[str]  #: map title
    uid: t.Optional[str]  #: unique id
    zoom: gws.gis.zoom.Config  #: map scales and resolutions


class Props(t.Data):
    crs: str
    crsDef: t.Optional[str]
    coordinatePrecision: int
    extent: t.Extent
    center: t.Point
    initResolution: float
    layers: t.List[t.ext.gis.layer.LayerProps]
    resolutions: t.List[float]
    title: str


class Object(gws.PublicObject, t.MapObject):
    def __init__(self):
        super().__init__()
        self.crs = ''
        self.extent = []
        self.center = []
        self.init_resolution = 0
        self.resolutions = []
        self.layers: t.List[t.LayerObject] = []

    def configure(self):
        super().configure()

        self.uid = self.var('uid') or 'map'
        p = self.get_closest('gws.common.project')
        if p:
            self.uid = p.uid + '.' + self.uid

        self.crs = self.var('crs')
        if not self.crs:
            raise ValueError('map requires a CRS')

        self.extent = self.var('extent')
        if not self.extent:
            raise ValueError('map requires an extent')

        self.center = self.var('center')
        if not self.center:
            self.center = [
                round(self.extent[0] + (self.extent[2] - self.extent[0]) / 2),
                round(self.extent[1] + (self.extent[3] - self.extent[1]) / 2),
            ]

        zoom = self.var('zoom')
        self.resolutions = gws.gis.zoom.effective_resolutions(zoom)
        self.init_resolution = gws.gis.zoom.init_resolution(zoom, self.resolutions)

        if not self.resolutions:
            raise ValueError('no resolutions for the map')

        for p in self.var('layers'):
            try:
                self.layers.append(self.add_child('gws.ext.gis.layer', p))
            except Exception:
                gws.log.exception()

    @property
    def props(self):
        proj = gws.gis.proj.as_proj(self.crs)
        prec = 7
        if proj.units == 'm':
            prec = 0
        return {
            'crs': proj.epsg,
            'crsDef': proj.proj4text,
            'coordinatePrecision': prec,
            'extent': self.extent,
            'center': self.center,
            'initResolution': self.init_resolution,
            'layers': self.layers,
            'resolutions': self.resolutions,
            'title': self.var('titie'),
        }
