import gws
import gws.types as t
import gws.gis.layer
import gws.gis.source
import gws.gis.legend
import gws.ows.wmts
import gws.ows.request
import gws.ows.util
import gws.gis.proj
import gws.gis.mpx
import gws.tools.misc as misc


class Config(gws.gis.layer.ProxiedConfig):
    """WMTS layer"""

    capsCacheMaxAge: t.duration = '1d'  #: max cache age for capabilities documents
    display: str = 'tile'
    format: str = 'image/jpeg'  #: image format
    layer: t.Optional[str]  #: WMTS layer name
    maxRequests: int = 1  #: max concurrent requests to this source
    params: t.Optional[dict]  #: query string parameters
    url: t.url  #: service url


class Object(gws.gis.layer.ProxiedTile):
    def __init__(self):
        super().__init__()

        self.layer: gws.ows.wmts.SourceLayer = None
        self.matrix_set = gws.ows.wmts.TileMatrixSet = None
        self.service: gws.ows.wmts.Service = None
        self.source_crs = ''
        self.url = ''

    def configure(self):
        super().configure()

        self.url = self.var('url')
        self.service = gws.ows.util.shared_service('WMTS', self, self.config)
        self.layer = self._get_layer(self.var('layer'))
        self.source_crs = gws.ows.util.best_crs(self.map.crs, self.service.supported_crs)
        self.matrix_set = self._get_matrix_set()

        # if no legend.url is given, use an auto legend

        if not self.legend_url and self.layer.legend:
            self.legend_url = self.layer.legend

        self.has_legend = self.var('legend.enabled') and bool(self.legend_url)

    def mapproxy_config(self, mc):
        m0 = self.matrix_set.matrices[0]
        res = [
            misc.scale2res(m.scale)
            for m in self.matrix_set.matrices
        ]

        src_grid_config = gws.compact({
            # nw = upper-left for WMTS
            'origin': 'nw',
            'bbox': m0.extent,
            'res': res,
            'srs': self.source_crs,
            'tile_size': [m0.tile_width, m0.tile_height],
        })

        src = self.mapproxy_back_cache_config(mc, self._get_tile_url(), src_grid_config)
        self.mapproxy_layer_config(mc, src)

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.service.meta,
        }
        return self.description_template.render(context).content

    def _get_layer(self, layer_name) -> gws.ows.wmts.SourceLayer:
        if not layer_name:
            return self.service.layers[0]
        for la in self.service.layers:
            if la.name == layer_name:
                return la
        raise gws.Error(f'layer {layer_name!r} not found')

    def _get_matrix_set(self):
        for matrix_set in self.layer.matrix_sets:
            if matrix_set.crs == self.source_crs:
                return matrix_set
        raise gws.Error(f'no suitable tile matrix set found')

    def _get_tile_url(self):
        url = gws.get(self.layer.resource_urls, 'tile')
        if url:
            url = url.replace('{TileMatrixSet}', self.matrix_set.uid)
            url = url.replace('{TileMatrix}', '%(z)02d')
            url = url.replace('{TileCol}', '%(x)d')
            url = url.replace('{TileRow}', '%(y)d')
            return url

        url = gws.get(self.service.operations, 'GetTile.get_url')
        if url:
            params = {
                'SERVICE': 'WMTS',
                'REQUEST': 'GetTile',
                'VERSION': self.service.version,
                'LAYER': self.layer.name,
                'FORMAT': self.layer.format or 'image/jpeg',
                'STYLE': self.var('style') or 'default',
                'TILEMATRIXSET': self.matrix_set.uid,
                'TILEMATRIX': '%(z)02d',
                'TILECOL': '%(x)d',
                'TILEROW': '%(y)d',
            }

            # NB cannot use as_query_string because of the MP's percent formatting
            qs = '&'.join(k + '=' + str(v or '') for k, v in params.items())
            return url + '?' + qs
