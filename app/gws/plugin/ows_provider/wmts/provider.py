"""WMTS provider"""

import gws
import gws.types as t
import gws.base.ows
import gws.base.metadata
import gws.lib.ows
import gws.lib.xml2
from . import caps


class Config(gws.base.ows.provider.Config):
    pass


class Object(gws.base.ows.provider.Object):
    protocol = gws.OwsProtocol.WMTS

    matrix_sets: t.List[caps.TileMatrixSet]
    source_layers: t.List[caps.SourceLayer]

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.matrix_sets = cc.matrix_sets
        self.metadata = t.cast(gws.IMetaData, self.create_child(gws.base.metadata.Object, cc.metadata))
        self.operations = cc.operations
        self.version = cc.version
        self.source_layers = cc.source_layers
        self.supported_crs = cc.supported_crs


##

def create(root: gws.RootObject, cfg: gws.Config, shared: bool = False, parent: gws.Object = None) -> Object:
    if not shared:
        return t.cast(Object, root.create_object(Object, cfg, parent))
    return t.cast(Object, root.create_shared_object(Object, cfg, uid=gws.sha256(cfg)))
