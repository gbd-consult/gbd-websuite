import gws
import gws.base.layer
import gws.gis.source
import gws.lib.net
import gws.gis.ows
import gws.lib.uom as units
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

        p = self.var('sourceGrid', default=gws.Config())
        crs = p.crs or gws.gis.crs.best_match(
            self.parentBounds.crs,
            gws.gis.source.combined_crs_list(self.sourceLayers))

        self.tileMatrixSet = self.get_tile_matrix_set_for_crs(crs)
        if not self.tileMatrixSet:
            raise gws.Error(f'no suitable tile matrix set found')

        extent = p.extent or self.tileMatrixSet.matrices[0].extent
        self.sourceGrid = gws.TileGrid(
            bounds=gws.Bounds(crs=crs, extent=extent),
            corner=p.corner or 'lt',
            tileSize=p.tileSize or self.tileMatrixSet.matrices[0].tileWidth,
        )
        self.sourceGrid.resolutions = (
                p.resolutions or
                sorted([units.scale_to_res(m.scale) for m in self.tileMatrixSet.matrices], reverse=True))

        p = self.var('targetGrid', default=gws.Config())
        self.targetGrid = gws.TileGrid(
            corner=p.corner or 'lt',
            tileSize=p.tileSize or 256,
        )
        crs = self.parentBounds.crs
        extent = (
            p.extent or
            self.sourceGrid.bounds.extent if crs == self.sourceGrid.bounds.crs else self.parentBounds.extent)
        self.targetGrid.bounds = gws.Bounds(crs=crs, extent=extent)
        self.targetGrid.resolutions = (
                p.resolutions or
                gws.gis.zoom.resolutions_from_bounds(self.targetGrid.bounds, self.targetGrid.tileSize))

        if not gws.base.layer.configure.bounds(self):
            self.bounds = self.parentBounds

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

    def render(self, lri):
        return gws.base.layer.util.generic_raster_render(self, lri)

    def mapproxy_config(self, mc):
        url = self.get_tile_url()

        # mapproxy encoding

        url = url.replace('{TileMatrix}', '%(z)02d')
        url = url.replace('{TileCol}', '%(x)d')
        url = url.replace('{TileRow}', '%(y)d')


        sg = self.sourceGrid

        if sg.corner == 'lt':
            origin = 'nw'
        elif sg.corner == 'lb':
            origin = 'sw'
        else:
            raise gws.Error(f'invalid grid corner {sg.corner!r}')

        back_grid_uid = mc.grid(gws.compact({
            'origin': origin,
            'srs': sg.bounds.crs.epsg,
            'bbox': sg.bounds.extent,
            'res': sg.resolutions,
            'tile_size': [sg.tileSize, sg.tileSize],
        }))

        src_uid = gws.base.layer.util.mapproxy_back_cache_config(self, mc, url, back_grid_uid)
        gws.base.layer.util.mapproxy_layer_config(self, mc, src_uid)
        return


        res = [units.scale_to_res(m.scale) for m in self.tileMatrixSet.matrices]
        m0 = self.tileMatrixSet.matrices[0]

        # res = [156543.03392804097, 78271.51696402048, 39135.75848201024, 19567.87924100512, 9783.93962050256, 4891.96981025128, 2445.98490512564, 1222.99245256282, 611.49622628141, 305.748113140705, 152.8740565703525, 76.43702828517625, 38.21851414258813, 19.109257071294063, 9.554628535647032, 4.777314267823516, 2.388657133911758, 1.194328566955879, 0.5971642834779395, 0.29858214173896974]

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
