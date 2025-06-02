"""Storage manager."""

import gws


class Config(gws.Config):
    """Storage configuration"""

    providers: list[gws.ext.config.storageProvider]
    """Storage providers."""


class Object(gws.StorageManager):
    def configure(self):
        self.providers = []

        for cfg in self.cfg('providers', default=[]):
            self.create_provider(cfg)

    def create_provider(self, cfg, **kwargs):
        prov = self.root.create_shared(gws.ext.object.storageProvider, cfg, **kwargs)

        # replace a provider with the same uid
        m = {p.uid: p for p in self.providers}
        m[prov.uid] = prov
        self.providers = list(m.values())

        return prov

    def find_provider(self, uid=None):
        if uid:
            for p in self.providers:
                if p.uid == uid:
                    return p

        elif self.providers:
            return self.providers[0]
