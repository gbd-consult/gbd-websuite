"""Core database utilities."""

import gws
import gws.gis.crs


class Config(gws.Config):
    """Database configuration"""

    providers: list[gws.ext.config.databaseProvider]
    """database providers"""


class Object(gws.DatabaseManager):
    providerMap: dict[str, gws.DatabaseProvider]

    def configure(self):
        self.providerMap = {}
        for cfg in self.cfg('providers', default=[]):
            prov = self.create_provider(cfg)
            self.providerMap[prov.uid] = prov

        self.register_middleware('db')

    ##

    def enter_middleware(self, req: gws.WebRequester):
        pass

    def exit_middleware(self, req: gws.WebRequester, res: gws.WebResponder):
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
