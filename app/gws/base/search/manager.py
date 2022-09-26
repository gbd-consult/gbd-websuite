import gws
import gws.types as t


class Config(gws.Config):
    finders: t.Optional[t.List[gws.ext.config.finder]]


class Object(gws.Node, gws.ISearchManager):

    def configure(self):
        self.finders = [self.create_child(gws.ext.object.finder, p) for p in self.var('finders', default=[])]

    def add_finder(self, config):
        self.finders.append(self.create_child(gws.ext.object.finder, config))
