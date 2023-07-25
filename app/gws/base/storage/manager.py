"""Storage manager."""

import gws
import gws.types as t


class Config(gws.Config):
    """Storage configuration"""

    providers: list[gws.ext.config.storageProvider]
    """storage providers"""


class Object(gws.Node, gws.IStorageManager):
    providers: dict[str, gws.IStorageProvider]

    def configure(self):
        self.providers = {}
        for p in self.cfg('providers', default=[]):
            prov = self.create_child(gws.ext.object.storageProvider, p)
            self.providers[prov.uid] = prov

    def provider(self, uid):
        return self.providers.get(uid)

    def first_provider(self):
        for prov in self.providers.values():
            return prov
