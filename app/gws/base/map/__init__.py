import gws
import gws.types as t
import gws.base.layer
import gws.lib.extent
import gws.lib.proj
import gws.lib.zoom
import gws.lib.units as units



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


class Config(gws.Config):
    """Map configuration"""

    center: t.Optional[gws.Point]  #: map center
    coordinatePrecision: t.Optional[int]  #: precision for coordinates
    crs: t.Optional[gws.Crs] = 'EPSG:3857'  #: crs for this map
    extent: t.Optional[gws.Extent]  #: map extent
    extentBuffer: t.Optional[int]  #: extent buffer
    layers: t.List[gws.ext.layer.Config]  #: collection of layers for this map
    skipInvalidLayers: bool = False  #: remove invalid layers from the map
    title: str = ''  #: map title
    zoom: t.Optional[gws.lib.zoom.Config]  #: map scales and resolutions


class Props(gws.Data):
    crs: str
    crsDef: t.Optional[str]
    coordinatePrecision: int
    extent: gws.Extent
    center: gws.Point
    initResolution: float
    layers: t.List[gws.base.layer.Props]
    resolutions: t.List[float]
    title: str = ''


class Object(gws.Node):
    layers: t.List[gws.ILayer]

    center: gws.Point
    coordinate_precision: int
    crs: gws.Crs
    extent: gws.Extent
    init_resolution: float
    resolutions: t.List[float]

    @property
    def bounds(self) -> gws.Bounds:
        return gws.Bounds(crs=self.crs, extent=self.extent)

    @property
    def props(self):
        proj = gws.lib.proj.as_proj(self.crs)
        return Props(
            crs=proj.epsg,
            crsDef=proj.proj4text,
            coordinatePrecision=self.coordinate_precision,
            extent=self.extent,
            center=self.center,
            initResolution=self.init_resolution,
            layers=self.layers,
            resolutions=self.resolutions,
            title=self.title,
        )

    def configure(self):
        uid = self.var('uid') or 'map'
        project = self.get_closest('gws.base.project')
        if project:
            uid = project.uid + '.' + uid
        self.set_uid(uid)

        self.crs = self.var('crs')
        self.title = self.var('title') or self.uid

        self.resolutions = _DEFAULT_RESOLUTIONS
        self.init_resolution = _DEFAULT_RESOLUTIONS[-1]

        zoom = self.var('zoom')
        if zoom:
            self.resolutions = gws.lib.zoom.resolutions_from_config(zoom)
            self.init_resolution = gws.lib.zoom.init_resolution(zoom, self.resolutions)

        self.layers = [t.cast(gws.ILayer, self.create_child('gws.ext.layer', c)) for c in self.var('layers')]

        self.extent = _configure_extent(self, self.crs, None)
        if not self.extent:
            raise gws.Error(f'no extent found for {self.uid!r}')
        _set_default_extent(self, self.extent)

        self.center = self.var('center') or gws.lib.extent.center(self.extent)

        self.coordinate_precision = self.var('coordinatePrecision')
        if self.coordinate_precision is None:
            proj = gws.lib.proj.as_proj(self.crs)
            self.coordinate_precision = 2 if proj.units == 'm' else 7


def _configure_extent(obj, target_crs, default_ext):
    layers = gws.get(obj, 'layers') or []

    # we have an explicit extent provided in the config

    config_ext = obj.var('extent')

    if config_ext:
        ext = gws.lib.extent.from_list(config_ext)
        if not ext:
            raise gws.Error(f'{obj.uid!r}: invalid extent {config_ext!r}')

        # configure sublayers using config_ext as a default
        for la in layers:
            _configure_extent(la, target_crs, ext)

        obj.extent = ext
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
            obj.extent = gws.lib.extent.merge(layer_ext_list)
        else:
            obj.extent = default_ext
        return obj.extent

    # obj is a leaf layer and has no configured extent
    # check if it has an own extent (from its source)

    own_bounds: gws.Bounds = gws.get(obj, 'own_bounds')

    if own_bounds:
        own_ext = own_bounds.extent
        buf = obj.var('extentBuffer', with_parent=True)
        if buf:
            own_ext = gws.lib.extent.buffer(own_ext, buf)
        own_ext = gws.lib.extent.transform(own_ext, own_bounds.crs, target_crs)
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
    for la in gws.get(obj, 'layers', []):
        _set_default_extent(la, extent)
