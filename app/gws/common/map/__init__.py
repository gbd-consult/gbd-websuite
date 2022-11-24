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
    @property
    def bounds(self) -> t.Bounds:
        return t.Bounds(crs=self.crs, extent=self.extent)

    def configure(self):
        super().configure()

        uid = self.var('uid') or 'map'
        p = t.cast(t.IProject, self.get_closest('gws.common.project'))
        if p:
            uid = p.uid + '.' + uid
        self.set_uid(uid)

        self.crs: t.Crs = self.var('crs')

        self.resolutions: t.List[float] = _DEFAULT_RESOLUTIONS
        self.init_resolution: float = _DEFAULT_RESOLUTIONS[-1]

        zoom = self.var('zoom')
        if zoom:
            self.resolutions = gws.gis.zoom.resolutions_from_config(zoom)
            self.init_resolution = gws.gis.zoom.init_resolution(zoom, self.resolutions)

        self.layers: t.List[t.ILayer] = gws.common.layer.add_layers_to_object(self, self.var('layers'))

        self.extent: t.Extent = _configure_extent(self, self.crs, None)
        if not self.extent:
            raise gws.Error(f'no extent found for {self.uid!r}')
        _set_default_extent(self, self.extent)

        self.center: t.Point = self.var('center') or gws.gis.extent.center(self.extent)

        self.coordinate_precision: float = self.var('coordinatePrecision')
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


def _configure_extent(obj, target_crs, default_ext):
    layers = gws.get(obj, 'layers') or []

    # we have an explicit extent provided in the config

    config_ext = obj.var('extent')

    if config_ext:
        if not gws.gis.extent.valid(config_ext):
            raise gws.Error(f'{obj.uid!r}: invalid extent {config_ext!r}')

        # configure sublayers using config_ext as a default
        for la in layers:
            _configure_extent(la, target_crs, config_ext)

        obj.extent = config_ext
        return obj.extent

    if layers:
        # no config extent, configure sublayers using the current default extent
        # set obj.extent to the sum of sublayers' extents

        layer_ext_list = []
        for la in layers:
            layer_ext = _configure_extent(la, target_crs, default_ext)
            if layer_ext:
                layer_ext_list.append(layer_ext)

        if layer_ext_list:
            obj.extent = gws.gis.extent.merge(layer_ext_list)
        else:
            obj.extent = default_ext
        return obj.extent

    # obj is a leaf layer and has no configured extent
    # check if it has an own extent (from its source)

    own_bounds: t.Bounds = gws.get(obj, 'own_bounds')

    if own_bounds:
        own_ext = own_bounds.extent
        buf = obj.var('extentBuffer', parent=True)
        if buf:
            own_ext = gws.gis.extent.buffer(own_ext, buf)
        own_ext = gws.gis.extent.transform(own_ext, own_bounds.crs, target_crs)
        obj.extent = own_ext
        return obj.extent

    # obj is a leaf layer and has neither configured nor own extent
    # try using the default extent

    if default_ext:
        obj.extent = default_ext
        return obj.extent

    # no extent can be computed, it will be set to the map extent later on
    return None


def _set_default_extent(obj, extent):
    if not gws.get(obj, 'extent'):
        obj.extent = extent
    layers = gws.get(obj, 'layers')
    if layers:
        for la in layers:
            _set_default_extent(la, extent)
