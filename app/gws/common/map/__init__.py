import gws
import gws.types as t

import gws.gis.proj
import gws.gis.zoom
import gws.gis.extent
import gws.common.layer
import gws.common.layer.types
import gws.tools.units as units

# https://wiki.openstreetmap.org/wiki/Zoom_levels

_DEFAULT_RESOLUTIONS = [
    units.scale2res(150 * 1e6),
    units.scale2res(70 * 1e6),
    units.scale2res(35 * 1e6),
    units.scale2res(15 * 1e6),
    units.scale2res(10 * 1e6),
    units.scale2res(4 * 1e6),
    units.scale2res(2 * 1e6),
    units.scale2res(1 * 1e6),
    units.scale2res(500 * 1e3),
    units.scale2res(250 * 1e3),
    units.scale2res(150 * 1e3),
    units.scale2res(70 * 1e3),
    units.scale2res(35 * 1e3),
    units.scale2res(15 * 1e3),
    units.scale2res(8 * 1e3),
    units.scale2res(4 * 1e3),
    units.scale2res(2 * 1e3),
    units.scale2res(1 * 1e3),
    units.scale2res(500),
]


class Config(t.Config):
    """Map configuration"""

    center: t.Optional[t.Point]  #: map center
    coordinatePrecision: t.Optional[int]  #: precision for coordinates
    crs: t.Optional[t.Crs] = 'EPSG:3857'  #: crs for this map
    extent: t.Optional[t.Extent]  #: map extent
    extentBuffer: t.Optional[int]  #: extent buffer
    layers: t.List[t.ext.layer.Config]  #: collection of layers for this map
    title: str = ''  #: map title
    skipInvalidLayers: bool = False  #: remove invalid layers from the map
    uid: t.Optional[str]  #: unique id
    zoom: t.Optional[gws.gis.zoom.Config]  #: map scales and resolutions


class Props(t.Data):
    crs: str
    crsDef: t.Optional[str]
    coordinatePrecision: int
    extent: t.Extent
    center: t.Point
    initResolution: float
    layers: t.List[gws.common.layer.types.LayerProps]
    resolutions: t.List[float]
    title: str = ''


#:export IMap
class Object(gws.Object, t.IMap):
    def __init__(self):
        super().__init__()
        self.crs: t.Crs = ''
        self.extent: t.Extent = []
        self.center: t.Point = []
        self.init_resolution = 0.0
        self.resolutions: t.List[float] = []
        self.layers: t.List[t.ILayer] = []
        self.coordinate_precision = 0

    @property
    def bounds(self) -> t.Bounds:
        return t.Bounds(crs=self.crs, extent=self.extent)

    def configure(self):
        super().configure()

        uid = self.var('uid') or 'map'
        p = self.get_closest('gws.common.project')
        if p:
            uid = p.uid + '.' + uid
        self.set_uid(uid)

        self.crs = self.var('crs')

        self.resolutions = _DEFAULT_RESOLUTIONS
        self.init_resolution = _DEFAULT_RESOLUTIONS[-1]

        zoom = self.var('zoom')
        if zoom:
            self.resolutions = gws.gis.zoom.resolutions_from_config(zoom)
            self.init_resolution = gws.gis.zoom.init_resolution(zoom, self.resolutions)

        self.layers = gws.common.layer.add_layers_to_object(self, self.var('layers'))

        self.extent = _configure_extent(self, self.crs, None)
        if not self.extent:
            raise gws.Error(f'no extent found for {self.uid!r}')
        _set_default_extent(self, self.extent)

        self.center = self.var('center') or gws.gis.extent.center(self.extent)

        self.coordinate_precision = self.var('coordinatePrecision')
        if self.coordinate_precision is None:
            proj = gws.gis.proj.as_proj(self.crs)
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


def _configure_extent(obj, target_crs, parent_explicit_extent):
    # explicit extent provided in the config

    ee = obj.var('extent')
    if ee and not gws.gis.extent.valid(ee):
        raise gws.Error(f'invalid extent {ee} for {obj.uid!r}')

    # if this is a group (or a map itself), configure extents for sublayers
    # using this (or parent's) explicit extent as default
    # and merge sublayers' extents unless there's an explicit ext.

    layers = gws.get(obj, 'layers')

    if layers:
        exts = []
        for la in layers:
            le = _configure_extent(la, target_crs, ee or parent_explicit_extent)
            if le:
                exts.append(le)
        if ee:
            obj.extent = ee
        elif exts:
            obj.extent = gws.gis.extent.merge(exts)
        return obj.extent

    # terminal layer, has an explicit extent - just use it

    if ee:
        obj.extent = ee
        return obj.extent

    # terminal layer, has its own extent and optionally an extent buffer

    own: t.Bounds = gws.get(obj, 'own_bounds')
    buf = obj.var('extentBuffer', parent=True)

    if own:
        oe = own.extent
        if buf:
            oe = gws.gis.extent.buffer(oe, buf)
        oe = gws.gis.extent.transform(oe, own.crs, target_crs)
        obj.extent = oe
        return obj.extent

    # terminal layer, use the parent's explicit extent

    if parent_explicit_extent:
        obj.extent = parent_explicit_extent
        return obj.extent

    raise gws.Error(f'cannot compute layer extent for {obj.uid!r}')


def _set_default_extent(obj, extent):
    if not getattr(obj, 'extent', None):
        obj.extent = extent
    for la in getattr(obj, 'layers', []):
        _set_default_extent(la, extent)
