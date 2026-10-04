"""Base raster layer."""

from typing import Optional, cast

import gws
import gws.base.grabber
import gws.gis.cache
import gws.lib.crs
import gws.lib.extent

from . import core

CACHE_NAME_LENGTH = 12


class Object(core.Object):
    """Base raster layer.

    Renders boxes and tiles through grabbers (see ``gws.base.grabber``), one
    per CRS supported by the application. Subclasses provide the grabber with
    ``create_grabber`` and the default cache name with ``create_cache_name``.
    """

    canRenderBox = True
    canRenderTile = True

    def post_configure(self):
        self.post_configure_grabbers()

    def post_configure_grabbers(self):
        """Create a grabber for each CRS supported by the application.

        The cache settings come from the ``cache`` configuration. The cache
        name defaults to ``create_cache_name`` and gets the CRS code appended.
        Caching is disabled (``maxAge`` 0) when ``withCache`` is off, and for
        CRSs not listed in ``cache.crs`` if that list is given. CRSs that are
        incompatible with the layer extent are skipped with a configuration
        warning.
        """
        if not (self.canRenderBox or self.canRenderTile):
            return

        p = cast(
            gws.gis.cache.LayerConfig,
            self.cfg('cache') or self.root.specs.read({}, 'gws.gis.cache.core.LayerConfig'),
        )
        cache_proto = gws.MapCache(
            name=p.name,
            maxAge=p.maxAge,
            maxLevel=p.maxLevel,
            requestBuffer=p.requestBuffer,
            requestTiles=p.requestTiles,
        )
        if not self.cfg('withCache'):
            cache_proto.maxAge = 0
        if not cache_proto.name:
            cache_proto.name = self.create_cache_name(cache_proto)

        cache_srids = []
        if p.crs:
            cache_srids = [gws.lib.crs.require(c).srid for c in p.crs]

        for crs in self.root.app.supported_crs():
            ext = crs.clip_wgs_extent(self.wgsExtent)
            if not ext:
                self.root.config_warning(f'extent {self.wgsExtent} is incompatible with {crs!r}')
                continue
            cache = gws.MapCache(**vars(cache_proto))
            if cache_srids and crs.srid not in cache_srids:
                cache.maxAge = 0
            cache.name += f'_{crs.srid}'

            opts = gws.base.grabber.Options(
                crs=crs,
                cache=cache,
                extent=gws.lib.extent.transform_from_wgs(ext, crs),
                imageFormat=self.imageFormat,
            )

            gr = self.create_grabber(opts)
            if gr:
                self.grabbers[crs.srid] = gr

    def create_cache_name(self, cache: gws.MapCache) -> str:
        """Create a default cache name for the layer.

        Subclasses must implement this.

        Args:
            cache: Cache settings.

        Returns:
            Cache name.

        Raises:
            ``NotImplementedError``: In the base class.
        """
        raise NotImplementedError(f'create_cache_name not implemented in {self!r}')

    def create_grabber(self, opts: gws.base.grabber.Options) -> Optional[gws.Grabber]:
        """Create a grabber for one CRS. Returns nothing in the base class.

        Args:
            opts: Grabber options: CRS, cache, extent and image format.

        Returns:
            A grabber, or ``None`` if the layer cannot serve this CRS.
        """
        pass

    ##

    def grabber_for(self, lri: gws.LayerRenderInput) -> Optional[gws.Grabber]:
        """Find the grabber for the target CRS of a render request.

        Args:
            lri: Render input; without ``targetCrs`` the map CRS is used.

        Returns:
            The grabber, or ``None`` if there is none for the CRS.
        """
        crs = lri.targetCrs or self.mapCrs
        return self.grabbers.get(crs.srid)

    def render_tile(self, lri):
        grabber = self.grabber_for(lri)
        if not grabber:
            return

        blob = grabber.get_tile_as_bytes((lri.x, lri.y, lri.z), lri.renderParams)
        return gws.LayerRenderOutput(content=blob)

    def render_box(self, lri):
        grabber = self.grabber_for(lri)
        if not grabber:
            return

        params = lri.renderParams
        w, h = lri.view.pxSize

        if not lri.view.rotation:
            blob = grabber.get_box_as_bytes(lri.view.bounds.extent, w, h, params)
            return gws.LayerRenderOutput(content=blob)

        circ = gws.lib.extent.circumsquare(lri.view.bounds.extent)
        d = gws.u.to_rounded_int(gws.lib.extent.diagonal((0, 0, w, h)))

        img = grabber.get_box_as_image(circ, d, d, params)
        img.rotate(-lri.view.rotation)
        img.crop(
            (
                d / 2 - w / 2,
                d / 2 - h / 2,
                d / 2 + w / 2,
                d / 2 + h / 2,
            )
        )
        blob = img.to_bytes(self.imageFormat.mimeTypes[0], self.imageFormat.options)
        return gws.LayerRenderOutput(content=blob)
