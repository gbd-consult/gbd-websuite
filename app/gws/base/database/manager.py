"""Core database utilities."""

from typing import cast

import gws
import gws.lib.crs


class Config(gws.Config):
    """Database configuration"""

    providers: list[gws.ext.config.databaseProvider]
    """database providers"""


class Object(gws.DatabaseManager):
    def configure(self):
        self.providers = []

        for cfg in self.cfg('providers', default=[]):
            self.create_provider(cfg)

        self.root.app.middlewareMgr.register(self, 'db')

        ##

    def enter_middleware(self, req: gws.WebRequester):
        pass

    def exit_middleware(self, req: gws.WebRequester, res: gws.WebResponder):
        # @TODO deinit providers
        pass

    ##

    def create_provider(self, cfg, **kwargs):
        prov = self.root.create_shared(gws.ext.object.databaseProvider, cfg, **kwargs)

        # replace a provider with the same uid
        m = {p.uid: p for p in self.providers}
        m[prov.uid] = prov
        self.providers = list(m.values())

        return prov

    def find_provider(self, uid=None, ext_type=None):
        if uid:
            for p in self.providers:
                if p.uid == uid and (not ext_type or p.extType == ext_type):
                    return p

        elif ext_type:
            for p in self.providers:
                if p.extType == ext_type:
                    return p

        elif self.providers:
            return self.providers[0]
