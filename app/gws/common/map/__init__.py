import gws
import gws.types as t

import gws.gis.proj
import gws.gis.zoom
import gws.gis.layer


class Config(t.Config):
    """Map configuration"""

    center: t.Optional[t.Point]  #: map center
    coordinatePrecision: t.Optional[int] #: precision for coordinates
    crs: t.Optional[t.crsref] = 'EPSG:3857'  #: crs for this map
    extent: t.Extent  #: map extent
    layers: t.List[t.ext.layer.Config]  #: collection of layers for this map
    title: str = ''  #: map title
    uid: t.Optional[str]  #: unique id
    zoom: t.Optional[gws.gis.zoom.Config]  #: map scales and resolutions


class Props(t.Data):
    crs: str
    crsDef: t.Optional[str]
    coordinatePrecision: int
    extent: t.Extent
    center: t.Point
    initResolution: float
    layers: t.List[gws.gis.layer.Props]
    resolutions: t.List[float]
    title: str = ''


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
        self.extent = self.var('extent')

        self.center = self.var('center')
        if not self.center:
            self.center = [
                round(self.extent[0] + (self.extent[2] - self.extent[0]) / 2),
                round(self.extent[1] + (self.extent[3] - self.extent[1]) / 2),
            ]

        self.resolutions = [1000, 1]
        self.init_resolution = 1000

        zoom = self.var('zoom')
        if zoom:
            self.resolutions = gws.gis.zoom.resolutions_from_config(zoom)
            self.init_resolution = gws.gis.zoom.init_resolution(zoom, self.resolutions)

        self.layers = gws.gis.layer.add_layers_to_object(self, self.var('layers'))

    @property
    def props(self):
        proj = gws.gis.proj.as_proj(self.crs)
        prec = self.var('coordinatePrecision')
        if prec is None:
            prec = 0 if proj.units == 'm' else 7
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
