import gws
import gws.common.layer
import gws.gis.source
import gws.gis.legend
import gws.gis.util
import gws.gis.proj
import gws.gis.mpx
import gws.tools.json2
import gws.tools.units as units

import gws.types as t

from . import types, provider


class Config(gws.common.layer.ImageTileConfig):
    """WMTS layer"""

    capsCacheMaxAge: t.Duration = '1d'  #: max cache age for capabilities documents
    maxRequests: int = 0  #: max concurrent requests to this source
    params: t.Optional[dict]  #: query string parameters
    sourceLayer: t.Optional[str]  #: WMTS layer name
    sourceStyle: str = 'default'  #: WMTS style name
    url: t.Url  #: service url


class Object(gws.common.layer.ImageTile):
    def __init__(self):
        super().__init__()

        self.matrix_set: types.TileMatrixSet = None
        self.provider: provider.Object = None
        self.source_layer: types.SourceLayer = None
        self.source_crs = ''
        self.url = ''

    def configure(self):
        super().configure()

        self.url = self.var('url')
        self.provider = gws.gis.util.shared_ows_provider(provider.Object, self, self.config)
        self.source_layer = self._get_layer(self.var('sourceLayer'))
        self.source_crs = gws.gis.util.best_crs(self.map.crs, self.provider.supported_crs)
        self.matrix_set = self._get_matrix_set()

        # if no legend.url is given, use an auto legend

        if not self.legend_url and self.source_layer.legend:
            self.legend_url = self.source_layer.legend

        self.has_legend = self.var('legend.enabled') and bool(self.legend_url)

    @property
    def own_extent(self):
        return gws.gis.source.extent_from_layers([self.source_layer], self.map.crs)

    def mapproxy_config(self, mc):
        m0 = self.matrix_set.matrices[0]
        res = [
            units.scale2res(m.scale)
            for m in self.matrix_set.matrices
        ]

        grid_uid = mc.grid(gws.compact({
            'origin': 'nw',  # nw = upper-left for WMTS
            'bbox': m0.extent,
            'res': res,
            'srs': self.source_crs,
            'tile_size': [m0.tile_width, m0.tile_height],
        }))

        src = self.mapproxy_back_cache_config(mc, self._get_tile_url(), grid_uid)
        self.mapproxy_layer_config(mc, src)

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.provider.meta,
        }
        return self.description_template.render(context).content

    def _get_layer(self, layer_name) -> types.SourceLayer:
        if not layer_name:
            return self.provider.source_layers[0]
        for la in self.provider.source_layers:
            if la.name == layer_name:
                return la
        raise gws.Error(f'layer {layer_name!r} not found')

    def _get_matrix_set(self):
        for matrix_set in self.source_layer.matrix_sets:
            if matrix_set.crs == self.source_crs:
                return matrix_set
        raise gws.Error(f'no suitable tile matrix set found')

    def _get_tile_url(self):
        url = gws.get(self.source_layer.resource_urls, 'tile')
        if url:
            url = url.replace('{TileMatrixSet}', self.matrix_set.uid)
            url = url.replace('{TileMatrix}', '%(z)02d')
            url = url.replace('{TileCol}', '%(x)d')
            url = url.replace('{TileRow}', '%(y)d')
            return url

        url = gws.get(self.provider.operations, 'GetTile.get_url')
        if url:
            params = {
                'SERVICE': 'WMTS',
                'REQUEST': 'GetTile',
                'VERSION': self.provider.version,
                'LAYER': self.source_layer.name,
                'FORMAT': self.source_layer.format or 'image/jpeg',
                'STYLE': self.var('sourceStyle'),
                'TILEMATRIXSET': self.matrix_set.uid,
                'TILEMATRIX': '%(z)02d',
                'TILECOL': '%(x)d',
                'TILEROW': '%(y)d',
            }

            # NB cannot use as_query_string because of the MP's percent formatting
            qs = '&'.join(k + '=' + str(v or '') for k, v in params.items())
            return url + '?' + qs
