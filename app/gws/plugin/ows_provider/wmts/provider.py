"""WMTS provider"""

import gws
import gws.types as t

from . import caps
from .. import core


class Config(core.ProviderConfig):
    pass


class Object(core.Provider):
    protocol = gws.OwsProtocol.WMTS

    tileMatrixSets: list[gws.TileMatrixSet]

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.metadata = cc.metadata
        self.sourceLayers = cc.sourceLayers
        self.version = cc.version
        self.tileMatrixSets = cc.tileMatrixSets

        self.configure_operations(cc.operations)
