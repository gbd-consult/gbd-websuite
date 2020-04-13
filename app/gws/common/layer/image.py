import PIL.Image
import io

import gws
import gws.gis.mpx as mpx
import gws.gis.extent

import gws.types as t

from . import layer, types


class Config(layer.Config):
    cache: types.CacheConfig = {}  #: cache configuration
    grid: types.GridConfig = {}  #: grid configuration
    imageFormat: types.ImageFormat = 'png8'  #: image format


class Image(layer.Layer):
    def configure(self):
        super().configure()

        self.can_render_box = True
        self.can_render_xyz = True
        self.supports_wms = True

    @property
    def props(self):
        p = super().props

        if self.display == 'tile':
            return gws.merge(p, {
                'type': 'tile',
                'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetXyz/layerUid/' + self.uid + '/z/{z}/x/{x}/y/{y}/t.png',
                'tileSize': self.grid.tileSize,
            })

        if self.display == 'box':
            return gws.merge(p, {
                'type': 'box',
                'url': gws.SERVER_ENDPOINT + '/cmd/mapHttpGetBox/layerUid/' + self.uid,
            })

        return p

    def render_box(self, rv, client_params=None):
        uid = self.uid
        if not self.has_cache:
            uid += '_NOCACHE'

        if not rv.rotation:
            return gws.gis.mpx.wms_request(uid, rv.bounds, rv.size_px[0], rv.size_px[1])

        # rotation: render a circumsquare around the wanted extent

        circ = gws.gis.extent.circumsquare(rv.bounds.extent)
        w, h = rv.size_px
        d = gws.gis.extent.diagonal((0, 0, w, h))

        r = gws.gis.mpx.wms_request(uid, t.Bounds(crs=rv.bounds.crs, extent=circ), d, d)
        if not r:
            return

        img: PIL.Image.Image = PIL.Image.open(io.BytesIO(r))

        # rotate the square (NB: PIL rotations are counter-clockwise)
        # and crop the square back to the wanted extent

        img = img.rotate(-rv.rotation, resample=PIL.Image.BICUBIC)
        img = img.crop((
            d / 2 - w / 2,
            d / 2 - h / 2,
            d / 2 + w / 2,
            d / 2 + h / 2,
        ))

        with io.BytesIO() as out:
            img.save(out, format='png')
            return out.getvalue()

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
            'srs': self.map.crs,
            'bbox': self.extent,
        }))

        meta_size = self.grid.metaSize or 4

        front_cache_config = {
            'sources': [source_uid],
            'grids': [self.grid_uid],
            'cache': {
                'type': 'file',
                'directory_layout': 'mp'
            },
            'meta_size': [meta_size, meta_size],
            'meta_buffer': self.grid.metaBuffer,
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
