"""WMTS provider"""

import gws
import gws.base.layer
import gws.base.ows
import gws.lib.uom
import gws.lib.net
import gws.types as t

from . import caps


class Config(gws.base.ows.ProviderConfig):
    grid: t.Optional[gws.base.layer.GridConfig]
    """source grid"""


class Object(gws.base.ows.provider.Object):
    protocol = gws.OwsProtocol.WMTS

    tileMatrixSets: list[gws.TileMatrixSet]
    grids: list[gws.TileGrid]

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.metadata = cc.metadata
        self.sourceLayers = cc.sourceLayers
        self.version = cc.version
        self.tileMatrixSets = cc.tileMatrixSets

        self.configure_grids()
        self.configure_operations(cc.operations)

    def configure_grids(self):
        self.grids = []

        p = self.cfg('grid')
        if p:
            self.grids.append(gws.TileGrid(
                bounds=gws.Bounds(crs=p.crs, extent=p.extent),
                origin=p.origin or gws.Origin.nw,
                tileSize=p.tileSize or self.tileMatrixSets[0].matrices[0].tileWidth,
            ))
            return True

        for tms in self.tileMatrixSets:
            grids = []
            if self.forceCrs:
                grids = [g for g in grids if g.bounds.crs == self.forceCrs]
                if not grids:
                    gws.Error(f'no TileMatrixSet found for {self.forceCrs!r} in {self}')
            self.grids = grids

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
    p = obj.cfg('provider')
    if p:
        return obj.root.create_shared(Object, p)
    p = obj.cfg('_defaultProvider')
    if p:
        return p
    raise gws.Error(f'no provider found for {obj!r}')
