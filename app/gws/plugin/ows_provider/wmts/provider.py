"""WMTS provider"""

import gws
import gws.types as t

from . import caps
from .. import core


class Config(core.ProviderConfig):
    pass


class Object(core.Provider):
    protocol = gws.OwsProtocol.WMTS

    tileMatrixSets: t.List[gws.TileMatrixSet]

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.tileMatrixSets = cc.tileMatrixSets
        self.metadata = cc.metadata
        self.operations = cc.operations
        self.version = cc.version
        self.source_layers = cc.source_layers
