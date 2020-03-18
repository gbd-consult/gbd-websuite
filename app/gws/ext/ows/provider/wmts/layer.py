import gws
import gws.common.layer
import gws.gis.source
import gws.gis.util
import gws.gis.mpx
import gws.gis.ows
import gws.tools.net
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
    sourceStyle: str = ''  #: WMTS style name
    url: t.Url  #: service url


class Object(gws.common.layer.ImageTile):
    def configure(self):
        super().configure()

        self.url = self.var('url')
        self.provider: provider.Object = gws.gis.ows.shared_provider(provider.Object, self, self.config)

        self.meta, self.title = self.configure_metadata(self.provider.meta)

        self.source_crs = gws.gis.util.best_crs(self.map.crs, self.provider.supported_crs)
        self.source_layer: types.SourceLayer = self._get_layer(self.var('sourceLayer'))
        self.matrix_set: types.TileMatrixSet = self._get_matrix_set()

        if not self.legend_url and self.source_layer.legend:
            self.legend_url = self.source_layer.legend
        self.has_legend = self.var('legend.enabled') and bool(self.legend_url)

        self.source_style = self.var('sourceStyle')

    @property
    def own_bounds(self):
        return gws.gis.source.bounds_from_layers([self.source_layer], self.source_crs)

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
        if layer_name:
            for sl in self.provider.source_layers:
                if sl.name == layer_name:
                    return sl
            raise gws.Error(f'layer {layer_name!r} not found')

        if self.provider.source_layers:
            return self.provider.source_layers[0]

        raise gws.Error(f'no layers found')

    def _get_matrix_set(self):
        for ms in self.source_layer.matrix_sets:
            if ms.crs == self.source_crs:
                return ms
        raise gws.Error(f'no suitable tile matrix set found')

    def _get_tile_url(self):
        url = gws.get(self.source_layer.resource_urls, 'tile')
        if url:
            url = url.replace('{TileMatrixSet}', self.matrix_set.uid)
            url = url.replace('{TileMatrix}', '%(z)02d')
            url = url.replace('{TileCol}', '%(x)d')
            url = url.replace('{TileRow}', '%(y)d')
            return url

        url = self.provider.operation('GetTile').get_url
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

            if self.source_style:
                params['STYLE'] = self.source_style

            p = gws.tools.net.parse_url(url)
            params.update(p['params'])

            # NB cannot use as_query_string because of the MP's percent formatting

            p['params'] = {}
            qs = '&'.join(k + '=' + str(v or '') for k, v in params.items())
            url = gws.tools.net.make_url(p) + '?' + qs

            return url

        raise gws.Error('no GetTile url found')
