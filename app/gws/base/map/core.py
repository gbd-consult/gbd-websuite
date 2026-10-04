"""Default map object."""

from typing import Optional

import gws
import gws.base.layer
import gws.lib.crs
import gws.lib.bounds
import gws.lib.extent
import gws.gis.zoom


@gws.ext.config.map('default')
class Config(gws.Config):
    """Map with its layers, extent, CRS and zoom levels."""

    center: Optional[gws.Point]
    """Initial map center."""
    coordinatePrecision: Optional[int]
    """Decimal places for coordinates."""
    crs: Optional[gws.CrsName] = 'EPSG:3857'
    """CRS of the map and its layers."""
    extent: Optional[gws.Extent]
    """Map extent in map CRS coordinates."""
    extentBuffer: Optional[int]
    """Buffer added around the configured extent, in map units."""
    layers: list[gws.ext.config.layer]
    """Map layers."""
    title: str = ''
    """Map title, used as the root layer title."""
    wrapX: bool = False
    """Repeat the world horizontally in the client."""
    zoom: Optional[gws.gis.zoom.Config]
    """Allowed scales or zoom levels and the initial zoom."""


@gws.ext.props.map('default')
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
    """Root layer of a map."""

    parent: 'Object'
    """The map."""


@gws.ext.object.map('default')
class Object(gws.Map):
    """Default map."""

    wrapX: bool
    """Repeat the world horizontally in the client."""

    def configure(self):
        self.title = self.cfg('title') or self.cfg('_defaultTitle') or ''

        p = self.cfg('crs')
        crs = gws.lib.crs.require(p) if p else gws.lib.crs.WEBMERCATOR

        p = self.cfg('extent')
        if p:
            ext = gws.lib.extent.from_list(p)
            if not ext or not gws.lib.extent.is_valid(ext):
                raise gws.ConfigurationError(f'invalid extent {p!r}')
            buf = self.cfg('extentBuffer') or 0
            self.bounds = gws.Bounds(crs=crs, extent=gws.lib.extent.buffer(ext, buf))
            self.wgsExtent = gws.lib.extent.transform_to_wgs(self.bounds.extent, crs)
        else:
            self.wgsExtent = crs.wgsMaxExtent
            self.bounds = gws.Bounds(crs=crs, extent=gws.lib.extent.transform_from_wgs(self.wgsExtent, crs))

        self.center = self.cfg('center') or gws.lib.extent.center(self.bounds.extent)
        self.wrapX = self.cfg('wrapX', default=False)

        p = self.cfg('zoom')
        gws.gis.zoom.warn_deprecated_options(p, self.root)
        self.resolutions = gws.gis.zoom.resolutions_from_config(p, crs=self.bounds.crs)
        self.initResolution = gws.gis.zoom.init_resolution(p, self.resolutions, crs=self.bounds.crs)

        p = self.cfg('coordinatePrecision')
        self.coordinatePrecision = p or self.bounds.crs.coordinatePrecision

        self.rootLayer = self.create_child(
            gws.ext.object.layer,
            type='group',
            title=self.title,
            layers=self.cfg('layers'),
            _parentWgsExtent=self.wgsExtent,
            _mapCrs=self.bounds.crs,
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
