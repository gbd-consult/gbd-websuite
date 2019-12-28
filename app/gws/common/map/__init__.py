import gws
import gws.types as t

import gws.gis.proj
import gws.gis.zoom
import gws.gis.extent
import gws.common.layer


class Config(t.Config):
    """Map configuration"""

    center: t.Optional[t.Point]  #: map center
    coordinatePrecision: t.Optional[int]  #: precision for coordinates
    crs: t.Optional[t.Crs] = 'EPSG:3857'  #: crs for this map
    extent: t.Optional[t.Extent]  #: map extent
    extentBuffer: t.Optional[int]  #: extent buffer
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
    layers: t.List[gws.common.layer.Props]
    resolutions: t.List[float]
    title: str = ''


class Object(gws.Object, t.MapObject):
    def __init__(self):
        super().__init__()
        self.crs = ''
        self.extent = []
        self.center = []
        self.init_resolution = 0
        self.resolutions = []
        self.layers: t.List[t.LayerObject] = []
        self.coordinate_precision = 0

    @property
    def auto_uid(self):
        return None

    def configure(self):
        super().configure()

        uid = self.var('uid') or 'map'
        p = self.get_closest('gws.common.project')
        if p:
            uid = p.uid + '.' + uid
        self.set_uid(uid)

        self.crs = self.var('crs')

        self.resolutions = [1000, 1]
        self.init_resolution = 1000

        zoom = self.var('zoom')
        if zoom:
            self.resolutions = gws.gis.zoom.resolutions_from_config(zoom)
            self.init_resolution = gws.gis.zoom.init_resolution(zoom, self.resolutions)

        self.layers = gws.common.layer.add_layers_to_object(self, self.var('layers'))

        _configure_extent(self, None)

        self.center = self.var('center')
        if not self.center:
            self.center = [
                round(self.extent[0] + (self.extent[2] - self.extent[0]) / 2),
                round(self.extent[1] + (self.extent[3] - self.extent[1]) / 2),
            ]

        proj = gws.gis.proj.as_proj(self.crs)
        self.coordinate_precision = self.var('coordinatePrecision')
        if self.coordinate_precision is None:
            self.coordinate_precision = 2 if proj.units == 'm' else 7


    @property
    def props(self):
        proj = gws.gis.proj.as_proj(self.crs)
        return Props({
            'crs': proj.epsg,
            'crsDef': proj.proj4text,
            'coordinatePrecision': self.coordinate_precision,
            'extent': self.extent,
            'center': self.center,
            'initResolution': self.init_resolution,
            'layers': self.layers,
            'resolutions': self.resolutions,
            'title': self.var('titie'),
        })


def _configure_extent(obj, parent_explicit_extent):
    # object extent provided in the config

    ee = obj.var('extent')
    if ee and not gws.gis.extent.valid(ee):
        raise gws.Error(f'invalid extent {ee} in {obj.uid!r}')

    # if this is a group (or a map itself), configure extents for sublayers
    # using this (or parent's) explicit extent as default
    # and merge sublayers' extents unless there's an explicit ext.

    layers = getattr(obj, 'layers', [])

    if layers:
        exts = []
        for la in layers:
            exts.append(_configure_extent(la, ee or parent_explicit_extent))
        obj.extent = ee or gws.gis.extent.merge(exts)
        return obj.extent

        # terminal layer, explicit extent - just use it

    if ee:
        obj.extent = ee
        return obj.extent

    # terminal layer, has an own extent and optionally an own or default extent buf.

    own = gws.gis.extent.valid(getattr(obj, 'own_extent', None))
    buf = obj.var('extentBuffer', parent=True)

    if own:
        if buf:
            own = gws.gis.extent.buffer(own, buf)
        obj.extent = own
        return obj.extent

    # terminal layer, use the parent's explicit extent

    if parent_explicit_extent:
        obj.extent = parent_explicit_extent
        return obj.extent

    # fail, an extent is required

    raise gws.Error(f'no extent found for {obj.uid!r}')
