import gws
import gws.types as t


class Config(gws.Config):
    """Database configuration"""

    providers: t.List[gws.ext.config.db]  #: database providers


class Object(gws.Node, gws.IDatabaseCollection):
    def configure(self):
        self.items = [self.root.create_shared(gws.ext.object.db, p) for p in self.var('providers', default=[])]
