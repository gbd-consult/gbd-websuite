import gws
import gws.gis.extent
import gws.lib.image
import gws.gis.mpx as mpx

from . import core, types


class Config(types.Config):
    cache: types.CacheConfig = {}  # type:ignore #: cache configuration
    grid: types.GridConfig = {}  # type:ignore #: grid configuration
    imageFormat: types.ImageFormat = types.ImageFormat.png8  #: image format


class Object(core.Object):
    """Base image layer"""

    can_render_box = True
    can_render_xyz = True
    supports_raster_ows = True

    def props_for(self, user):
        p = super().props_for(user)

        if self.display == 'tile':
            return gws.merge(
                p,
                type='tile',
                url=core.layer_url_path(self.uid, kind='tile'),
                tileSize=self.grid.tileSize,
            )

        if self.display == 'box':
            return gws.merge(
                p,
                type='box',
                url=core.layer_url_path(self.uid, kind='box'),
            )

        return p

    def render_box(self, view, extra_params=None):
        uid = self.uid
        if not self.has_cache:
            uid += '_NOCACHE'

        if not view.rotation:
            return gws.gis.mpx.wms_request(uid, view.bounds, view.size_px[0], view.size_px[1], forward=extra_params)

        # rotation: render a circumsquare around the wanted extent

        circ = gws.gis.extent.circumsquare(view.bounds.extent)
        w, h = view.size_px
        d = gws.gis.extent.diagonal((0, 0, w, h))

        r = gws.gis.mpx.wms_request(
            uid,
            gws.Bounds(crs=view.bounds.crs, extent=circ),
            width=d,
            height=d,
            forward=extra_params)
        if not r:
            return

        img = gws.lib.image.from_bytes(r)

        # rotate the square (NB: PIL rotations are counter-clockwise)
        # and crop the square back to the wanted extent

        img.rotate(-view.rotation).crop((
            d / 2 - w / 2,
            d / 2 - h / 2,
            d / 2 + w / 2,
            d / 2 + h / 2,
        ))

        return img.to_bytes()

    def render_xyz(self, x, y, z):
        return gws.gis.mpx.wmts_request(
            self.uid,
            x, y, z,
            tile_matrix=self.grid_uid,
            tile_size=self.grid.tileSize)

    """
        Mapproxy config is done in two steps

        1. first, configure the source. For box layers, this is a normal WMS source.
        For tiled layers, we use the 'double cache' technique, see

        https://mapproxy.org/docs/nightly/configuration_examples.html#create-wms-from-existing-tile-server
        https://mapproxy.org/docs/1.11.0/configuration_examples.html#reprojecting-tiles

        Basically, the source is wrapped in a no-store BACK cache, which is then given to the front mpx layer

        2. then, configure the layer. Create the FRONT cache, which is store or no-store, depending on the cache setting.
        Also, configure the _NOCACHE variant for the layer, which skips the DST cache
    """

    def mapproxy_layer_config(self, mc, source_uid):

        mc.layer({
            'name': self.uid + '_NOCACHE',
            'sources': [source_uid]
        })

        res = [r for r in self.resolutions if r]
        if len(res) < 2:
            res = [res[0], res[0]]

        self.grid_uid = mc.grid(gws.compact({
            'origin': self.grid.origin,
            'tile_size': [self.grid.tileSize, self.grid.tileSize],
            'res': res,
            'srs': self.map.crs.epsg,
            'bbox': self.extent,
        }))

        meta_size = self.grid.reqSize or 4

        front_cache_config = {
            'sources': [source_uid],
            'grids': [self.grid_uid],
            'cache': {
                'type': 'file',
                'directory_layout': 'mp'
            },
            'meta_size': [meta_size, meta_size],
            'meta_buffer': self.grid.reqBuffer,
            'disable_storage': not self.has_cache,
            'minimize_meta_requests': True,
            'format': self.image_format,
        }

        self.cache_uid = mc.cache(front_cache_config)

        mc.layer({
            'name': self.uid,
            'sources': [self.cache_uid]
        })

    def mapproxy_back_cache_config(self, mc, url, grid_uid):
        source_uid = mc.source({
            'type': 'tile',
            'url': url,
            'grid': grid_uid,
            'concurrent_requests': self.var('maxRequests', default=0)
        })

        return mc.cache(gws.compact({
            'sources': [source_uid],
            'grids': [grid_uid],
            'cache': {
                'type': 'file',
                'directory_layout': 'mp'
            },
            'disable_storage': True,
            'format': self.image_format,
        }))
