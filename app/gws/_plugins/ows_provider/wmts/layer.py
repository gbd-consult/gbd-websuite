import gws
import gws.types as t
import gws.base.layer
import gws.lib.mpx
import gws.lib.ows
import gws.lib.source
import gws.lib.gisutil
import gws.lib.json2
import gws.lib.net
import gws.lib.units as units
from . import types, provider


class Config(gws.base.layer.ImageTileConfig):
    """WMTS layer"""

    capsCacheMaxAge: gws.Duration = '1d'  #: max cache age for capabilities documents
    maxRequests: int = 0  #: max concurrent requests to this source
    params: t.Optional[dict]  #: query string parameters
    sourceLayer: t.Optional[str]  #: WMTS layer name
    sourceStyle: str = ''  #: WMTS style name
    url: gws.Url  #: service url


class Object(gws.base.layer.ImageTile):
    def configure(self):
        

        self.url = self.var('url')
        self.provider: provider.Object = gws.lib.ows.shared_provider(provider.Object, self, self.config)

        self.meta = self.configure_metadata(self.provider.meta)
        self.title = self.meta.title

        self.source_crs = gws.lib.gisutil.best_crs(self.map.crs, self.provider.supported_crs)
        self.source_layer: types.SourceLayer = self._get_layer(self.var('sourceLayer'))
        self.matrix_set: types.TileMatrixSet = self._get_matrix_set()

        self.source_style = self.var('sourceStyle')

    @property
    def own_bounds(self):
        return gws.lib.source.bounds_from_layers([self.source_layer], self.source_crs)

    def configure_legend(self):
        return super().configure_legend() or gws.LayerLegend(
            enabled=bool(self.source_layer.legend),
            url=self.source_layer.legend)

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

            p = gws.lib.net.parse_url(url)
            params.update(p['params'])

            # NB cannot use as_query_string because of the MP's percent formatting

            p['params'] = {}
            qs = '&'.join(k + '=' + str(v or '') for k, v in params.items())
            url = gws.lib.net.make_url(p) + '?' + qs

            return url

        raise gws.Error('no GetTile url found')
