import gws
import gws.base.layer
import gws.lib.gis
import gws.lib.net
import gws.lib.units as units
import gws.types as t

from . import provider as provider_module


@gws.ext.Config('layer.wmts')
class Config(gws.base.layer.image.Config, provider_module.Config):
    """WMTS layer"""
    display: gws.base.layer.types.DisplayMode = gws.base.layer.types.DisplayMode.tile  #: layer display mode
    sourceLayer: t.Optional[str]  #: WMTS layer name
    sourceStyle: str = ''  #: WMTS style name


@gws.ext.Object('layer.wmts')
class Object(gws.base.layer.image.Object):
    matrix_set: gws.lib.gis.TileMatrixSet
    provider: provider_module.Object
    source_crs: gws.Crs
    source_layer: gws.lib.gis.SourceLayer
    source_style: str

    def configure(self):
        pass

    def configure_source(self):
        self.provider = self.root.create_object(provider_module.Object, self.config, shared=True)

        self.grid.reqSize = self.grid.reqSize or 1

        self.source_crs = self.provider.source_crs or gws.lib.gis.best_crs(self.map.crs, self.provider.supported_crs)
        self.source_layer = self.get_source_layer(self.var('sourceLayer'))
        self.matrix_set = self.get_matrix_set_for_crs(self.source_crs)
        self.source_style = self.var('sourceStyle')

        return True

    def configure_metadata(self):
        if not super().configure_metadata():
            self.set_metadata(self.provider.metadata)
            return True

    def configure_legend(self):
        if not super().configure_legend():
            if self.source_layer.legend_url:
                self.legend = gws.Legend(
                    enabled=True,
                    urls=[self.source_layer.legend_url],
                    cache_max_age=self.var('legend.cacheMaxAge', default=0),
                    options=self.var('legend.options', default={}))
                return True

    @property
    def own_bounds(self):
        return gws.lib.gis.bounds_from_source_layers([self.source_layer], self.source_crs)

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.provider.metadata,
        }
        return self.description_template.render(context).content

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

        src = self.mapproxy_back_cache_config(mc, self.get_tile_url(), grid_uid)
        self.mapproxy_layer_config(mc, src)

    def get_source_layer(self, layer_name) -> gws.lib.gis.SourceLayer:
        if layer_name:
            for sl in self.provider.source_layers:
                if sl.name == layer_name:
                    return sl
            raise gws.Error(f'layer {layer_name!r} not found')

        if self.provider.source_layers:
            return self.provider.source_layers[0]

        raise gws.Error(f'no layers found')

    def get_matrix_set_for_crs(self, crs):
        for ms in self.source_layer.matrix_sets:
            if ms.crs == crs:
                return ms
        raise gws.Error(f'no suitable tile matrix set found')

    def get_tile_url(self):
        resource_url = gws.get(self.source_layer, 'resource_urls.tile')

        if resource_url:
            return (resource_url
                    .replace('{TileMatrixSet}', self.matrix_set.uid)
                    .replace('{TileMatrix}', '%(z)02d')
                    .replace('{TileCol}', '%(x)d')
                    .replace('{TileRow}', '%(y)d'))

        operation = self.provider.operation(gws.OwsVerb.GetTile)

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

        pu = gws.lib.net.parse_url(operation.get_url)
        params.update(pu.params)

        # NB cannot use as_query_string because of the MP's percent formatting

        qs = '&'.join(k + '=' + str(v or '') for k, v in params.items())

        return gws.lib.net.make_url(pu, params={}) + '?' + qs
