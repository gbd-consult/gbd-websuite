import gws
import gws.base.layer
import gws.gis.source
import gws.lib.net
import gws.gis.ows
import gws.lib.units as units
import gws.types as t

from . import provider


@gws.ext.config.layer('wmts')
class Config(gws.base.layer.Config, provider.Config):
    """WMTS layer"""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.tile  #: layer display mode
    sourceLayer: t.Optional[str]  #: WMTS layer name


@gws.ext.object.layer('wmts')
class Object(gws.base.layer.Object, gws.IOwsClient):
    provider: provider.Object
    tileMatrixSet: gws.TileMatrixSet
    sourceCrs: gws.ICrs
    styleName: str

    def configure(self):
        self.provider = self.var('_provider') or self.root.create_shared(provider.Object, self.config)

        self.sourceLayers = self.var('_sourceLayers')
        if not self.sourceLayers:
            slf = gws.merge(
                gws.gis.source.LayerFilter(isImage=True),
                self.var('sourceLayers'))
            self.sourceLayers = gws.gis.source.filter_layers(self.provider.sourceLayers, slf)
        if not self.sourceLayers:
            raise gws.Error(f'no source layers found in {self.provider.url!r}')

        if not gws.base.layer.configure.metadata(self):
            self.metadata = self.provider.metadata

        self.sourceCrs = gws.gis.crs.best_match(
            self.provider.forceCrs or self.parentBounds.crs,
            gws.gis.source.combined_crs_list(self.sourceLayers))

        self.tileMatrixSet = self.get_tile_matrix_set_for_crs(self.sourceCrs)
        if not self.tileMatrixSet:
            raise gws.Error(f'no suitable tile matrix set found')

        if not gws.base.layer.configure.bounds(self):
            crs = self.parentBounds.crs
            self.bounds = gws.Bounds(
                crs=crs,
                extent=gws.gis.extent.transform(
                    self.tileMatrixSet.matrices[0].extent,
                    self.tileMatrixSet.crs,
                    crs))

        if not gws.base.layer.configure.resolutions(self):
            res = [units.scale_to_res(m.scale) for m in self.tileMatrixSet.matrices]
            self.resolutions = sorted(res, reverse=True)

        if not gws.base.layer.configure.legend(self):
            url = self.sourceLayers[0].legendUrl
            if url:
                self.legend = self.create_child(
                    gws.ext.object.legend,
                    gws.Config(type='remote', urls=[url]))

        self.styleName = ''
        if self.sourceLayers[0].defaultStyle:
            self.styleName = self.sourceLayers[0].defaultStyle.name

        self.grid.reqSize = self.grid.reqSize or 1

    def render(self, lri):
        return gws.base.layer.util.generic_raster_render(self, lri)

    def mapproxy_config(self, mc):
        res = [units.scale_to_res(m.scale) for m in self.tileMatrixSet.matrices]
        m0 = self.tileMatrixSet.matrices[0]

        grid_uid = mc.grid(gws.compact({
            'origin': 'nw',  # nw = upper-left for WMTS
            'bbox': m0.extent,
            'res': res,
            'srs': self.sourceCrs.epsg,
            'tile_size': [m0.tileWidth, m0.tileHeight],
        }))

        url = self.get_tile_url()

        # mapproxy encoding

        url = url.replace('{TileMatrix}', '%(z)02d')
        url = url.replace('{TileCol}', '%(x)d')
        url = url.replace('{TileRow}', '%(y)d')

        src_uid = gws.base.layer.util.mapproxy_back_cache_config(self, mc, url, grid_uid)
        gws.base.layer.util.mapproxy_layer_config(self, mc, src_uid)

    def get_tile_matrix_set_for_crs(self, crs):
        for tms in self.sourceLayers[0].tileMatrixSets:
            if tms.crs == crs:
                return tms

    def get_tile_url(self):
        ru = self.sourceLayers[0].resourceUrls
        resource_url = ru.get('tile') if ru else None

        if resource_url:
            return (
                resource_url
                .replace('{TileMatrixSet}', self.tileMatrixSet.uid)
                .replace('{Style}', self.styleName or 'default'))

        params = {
            'SERVICE': 'WMTS',
            'REQUEST': 'GetTile',
            'VERSION': self.provider.version,
            'LAYER': self.sourceLayers[0].name,
            'FORMAT': self.sourceLayers[0].imageFormat or 'image/jpeg',
            'TILEMATRIXSET': self.tileMatrixSet.uid,
            'TILEMATRIX': '{TileMatrix}',
            'TILECOL': '{TileCol}',
            'TILEROW': '{TileRow}',
        }

        if self.styleName:
            params['STYLE'] = self.styleName

        op = self.provider.operation(gws.OwsVerb.GetTile)
        args = self.provider.request_args_for_operation(op, params=params)
        url = gws.lib.net.add_params(args.url, args.params)

        # {} should not be encoded
        return url.replace('%7B', '{').replace('%7D', '}')
