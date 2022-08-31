import gws
import gws.types as t


class Config(gws.Config):
    finders: t.Optional[t.List[gws.ext.config.finder]]


class Object(gws.Node, gws.IFinderCollection):
    def configure(self):
        self.items = [self.create_child(gws.ext.object.finder, p) for p in self.var('finders', default=[])]

    def add_finder(self, config):
        self.items.append(self.create_child(gws.ext.object.finder, config))
