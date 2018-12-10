import gws.ows.wmts
import gws.ows.util
import gws.gis.zoom
import gws.config
import gws.tools.net

import gws.tools.misc as misc
import gws.types as t
import gws.gis.source
import gws.gis.proj


class Config(gws.gis.source.BaseConfig):
    """WMTS source"""

    capsCacheMaxAge: t.duration = '1d'  #: max cache age for capabilities documents
    format: str = 'image/jpeg'  #: image format
    layer: t.Optional[str]  #: WMTS layer name
    options: t.Optional[dict]  #: additional options
    style: str = ''  #: image style
    url: t.url  #: service url


class Object(gws.gis.source.Base, t.SourceObject):
    def __init__(self):
        super().__init__()
        self.service: gws.ows.wmts.Service = None
        self.layer: gws.ows.wmts.SourceLayer = None
        self.tms: gws.ows.wmts.TileMatrixSet = None
        self.crs = ''
        self.url = ''

    def configure(self):
        super().configure()

        self.url = self.var('url')
        self.service = gws.ows.util.shared_service('WMTS', self, self.config)
        self.layer = self._get_layer(self.var('layer'))
        self.crs = gws.ows.util.crs_for_object(self, self.service.supported_crs)
        self.tms = self._get_tms()

        if not self.tms:
            raise gws.config.LoadError(f'no suitable tile matrix set found')

        self.extent = self.tms.matrices[0].extent

    def _get_layer(self, layer_name) -> gws.ows.wmts.SourceLayer:
        if not layer_name:
            return self.service.layers[0]
        for la in self.service.layers:
            if la.name == layer_name:
                return la
        raise gws.config.LoadError(f'layer {layer_name!r} not found')

    def _get_tms(self):
        for tms in self.layer.matrix_sets:
            if tms.crs == self.crs:
                return tms

    def mapproxy_config(self, mc, options=None):
        # we use the same double-cache as with tile sources
        # see also:
        # https://mapproxy.org/docs/nightly/configuration_examples.html#create-wms-from-existing-tile-server

        # @TODO: merge with source/tile

        params = {
            'SERVICE': 'WMTS',
            'REQUEST': 'GetTile',
            'VERSION': self.service.version,
            'LAYER': self.layer.name,
            'FORMAT': self.layer.format or 'image/jpeg',
            'STYLE': self.var('style') or 'default',
            'TILEMATRIXSET': self.tms.uid,
            'TILEMATRIX': '%(z)02d',
            'TILEROW': '%(y)d',
            'TILECOL': '%(x)d',
        }

        url = self.service.operations['GetTile'].get_url

        # NB cannot use as_query_string because of the MP's percent formatting
        qs = '&'.join(k + '=' + str(v or '') for k, v in params.items())
        url += '?' + qs

        m0 = self.tms.matrices[0]
        res = [
            misc.scale2res(m.scale)
            for m in self.tms.matrices
        ]

        src_grid_config = gws.compact({
            # nw = upper-left for WMTS
            'origin': 'nw',
            'bbox': m0.extent,
            'res': res,
            'srs': self.crs,
            'tile_size': [m0.tile_width, m0.tile_height],
        })

        src_grid = mc.grid(self, src_grid_config, self.uid + '_src')

        source = mc.source(self, {
            'type': 'tile',
            'url': url,
            'grid': src_grid,
        })

        src_cache_options = {
            'type': 'file',
            'directory_layout': 'mp'
        }

        # NB for tiled sources always request one tile at a time

        src_cache_config = gws.compact({
            'sources': [source],
            'grids': [src_grid],
            'cache': src_cache_options,
            'disable_storage': True,
            'meta_size': [1, 1],
            'meta_buffer': 0,
            'minimize_meta_requests': True,
        })

        src_cache = mc.cache(self, src_cache_config, self.uid + '_src')
        return src_cache
