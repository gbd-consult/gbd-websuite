"""Core database utilities."""

import gws
import gws.gis.crs


class Config(gws.Config):
    """Database configuration"""

    providers: list[gws.ext.config.databaseProvider]
    """database providers"""


class Object(gws.Node, gws.IDatabaseManager):
    providerMap: dict[str, gws.IDatabaseProvider]

    def configure(self):
        self.providerMap = {}
        for cfg in self.cfg('providers', default=[]):
            prov = self.create_provider(cfg)
            self.providerMap[prov.uid] = prov

        self.root.app.register_middleware('db', self)

    ##

    def enter_middleware(self, req: gws.IWebRequester):
        pass

    def exit_middleware(self, req: gws.IWebRequester, res: gws.IWebResponder):
        # @TODO deinit providers
        pass

    ##

    def create_provider(self, cfg, **kwargs):
        prov = self.root.create_shared(gws.ext.object.databaseProvider, cfg, _defaultManager=self, **kwargs)
        self.providerMap[prov.uid] = prov
        return prov

    def providers(self):
        return list(self.providerMap.values())

    def provider(self, uid):
        return self.providerMap.get(uid)

    def first_provider(self, ext_type: str):
        for prov in self.providerMap.values():
            if prov.extType == ext_type:
                return prov
