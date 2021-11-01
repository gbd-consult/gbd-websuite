import gws
import gws.base.layer
import gws.lib.gis.source
import gws.lib.gis.util
import gws.lib.net
import gws.lib.units as units
import gws.types as t

from . import provider as provider_module


@gws.ext.Config('layer.wmts')
class Config(gws.base.layer.image.Config, provider_module.Config):
    """WMTS layer"""
    display: gws.base.layer.types.DisplayMode = gws.base.layer.types.DisplayMode.tile  #: layer display mode
    sourceLayer: t.Optional[str]  #: WMTS layer name


@gws.ext.Object('layer.wmts')
class Object(gws.base.layer.image.Object, gws.IOwsClient):
    provider: provider_module.Object
    tile_matrix_set: gws.TileMatrixSet
    source_layer: gws.SourceLayer
    source_crs: gws.ICrs
    style_name: str

    def configure_source(self):
        gws.lib.gis.util.configure_ows_client_layers(self, provider_module.Object)

        self.source_crs = gws.lib.gis.util.best_crs(
            self.provider.force_crs or self.crs,
            gws.lib.gis.source.supported_crs_list(self.source_layers))

        if len(self.source_layers) > 1:
            gws.log.warn(f'multiple layers found for {self.uid!r}, using the first one')
        self.source_layer = self.source_layers[0]

        self.tile_matrix_set = self.get_tile_matrix_set_for_crs(self.source_crs)
        if not self.tile_matrix_set:
            raise gws.Error(f'no suitable tile matrix set found for layer={self.uid!r}')

        self.style_name = ''
        if self.source_layer.default_style:
            self.style_name = self.source_layer.default_style.name

        self.grid.reqSize = self.grid.reqSize or 1

        return True

    def configure_metadata(self):
        if not super().configure_metadata():
            self.set_metadata(self.provider.metadata)
            return True

    def configure_zoom(self):
        if not super().configure_zoom():
            return gws.lib.gis.util.configure_ows_client_zoom(self)

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
        return gws.lib.gis.source.combined_bounds(self.source_layers, self.source_crs)

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.provider.metadata,
        }
        return self.description_template.render(context).content

    def mapproxy_config(self, mc):
        m0 = self.tile_matrix_set.matrices[0]
        res = [
            units.scale2res(m.scale)
            for m in self.tile_matrix_set.matrices
        ]

        grid_uid = mc.grid(gws.compact({
            'origin': 'nw',  # nw = upper-left for WMTS
            'bbox': m0.extent,
            'res': res,
            'srs': self.source_crs.epsg,
            'tile_size': [m0.tile_width, m0.tile_height],
        }))

        src = self.mapproxy_back_cache_config(mc, self.get_tile_url(), grid_uid)
        self.mapproxy_layer_config(mc, src)

    def get_tile_matrix_set_for_crs(self, crs):
        for tms in self.source_layer.tile_matrix_sets:
            if tms.crs == crs:
                return tms

    def get_tile_url(self):
        resource_url = gws.get(self.source_layer, 'resource_urls.tile')

        if resource_url:
            return (resource_url
                    .replace('{TileMatrixSet}', self.tile_matrix_set.uid)
                    .replace('{TileMatrix}', '%(z)02d')
                    .replace('{TileCol}', '%(x)d')
                    .replace('{TileRow}', '%(y)d')
                    .replace('{Style}', self.style_name or 'default'))

        operation = self.provider.operation(gws.OwsVerb.GetTile)

        params = {
            'SERVICE': 'WMTS',
            'REQUEST': 'GetTile',
            'VERSION': self.provider.version,
            'LAYER': self.source_layer.name,
            'FORMAT': self.source_layer.image_format or 'image/jpeg',
            'TILEMATRIXSET': self.tile_matrix_set.uid,
            'TILEMATRIX': '%(z)02d',
            'TILECOL': '%(x)d',
            'TILEROW': '%(y)d',
        }

        if self.style_name:
            params['STYLE'] = self.style_name

        pu = gws.lib.net.parse_url(operation.get_url)
        params.update(pu.params)

        # NB cannot use as_query_string because of the MP's percent formatting

        qs = '&'.join(k + '=' + str(v or '') for k, v in params.items())

        return gws.lib.net.make_url(pu, params={}) + '?' + qs
