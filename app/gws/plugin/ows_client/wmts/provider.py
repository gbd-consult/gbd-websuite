"""WMTS provider"""

import gws
import gws.base.layer
import gws.base.ows.client
import gws.config.util
import gws.lib.uom
import gws.lib.net
import gws.types as t

from . import caps


class Config(gws.base.ows.client.provider.Config):
    grid: t.Optional[gws.base.layer.GridConfig]
    """source grid"""


class Object(gws.base.ows.client.provider.Object):
    protocol = gws.OwsProtocol.WMTS

    tileMatrixSets: list[gws.TileMatrixSet]
    grids: list[gws.TileGrid]

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.metadata = cc.metadata
        self.sourceLayers = cc.sourceLayers
        self.version = cc.version
        self.tileMatrixSets = cc.tileMatrixSets

        self.configure_operations(cc.operations)

    def grid_for_tms(self, tms: gws.TileMatrixSet) -> gws.TileGrid:
        return gws.TileGrid(
            bounds=gws.Bounds(crs=tms.crs, extent=tms.matrices[0].extent),
            origin=gws.Origin.nw,
            resolutions=sorted([gws.lib.uom.scale_to_res(m.scale) for m in tms.matrices], reverse=True),
            tileSize=tms.matrices[0].tileWidth,
        )

    def tile_url_template(self, sl: gws.SourceLayer, tms: gws.TileMatrixSet, style: gws.SourceStyle) -> str:
        ru = sl.resourceUrls
        resource_url = ru.get('tile') if ru else None

        if resource_url:
            return (
                resource_url
                .replace('{TileMatrixSet}', tms.uid)
                .replace('{Style}', style.name))

        params = {
            'SERVICE': gws.OwsProtocol.WMTS,
            'REQUEST': gws.OwsVerb.GetTile,
            'VERSION': self.version,
            'LAYER': sl.name,
            'FORMAT': sl.imageFormat or 'image/jpeg',
            'TILEMATRIXSET': tms.uid,
            'STYLE': style.name,
            'TILEMATRIX': '{TileMatrix}',
            'TILECOL': '{TileCol}',
            'TILEROW': '{TileRow}',
        }

        op = self.get_operation(gws.OwsVerb.GetTile)
        args = self.prepare_operation(op, params=params)
        url = gws.lib.net.add_params(args.url, args.params)

        # {} should not be encoded
        return url.replace('%7B', '{').replace('%7D', '}')


#

def get_for(obj: gws.INode) -> Object:
    return t.cast(Object, gws.config.util.get_provider(Object, obj))
