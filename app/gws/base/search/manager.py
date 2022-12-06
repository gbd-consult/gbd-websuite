import gws
import gws.types as t


class Config(gws.Config):
    finders: t.Optional[t.List[gws.ext.config.finder]]


class Object(gws.Node, gws.ISearchManager):

    def configure(self):
        self.finders = [self.create_child(gws.ext.object.finder, p) for p in self.var('finders', default=[])]

    def add_finder(self, f):
        self.finders.append(f)

    def get_finder_for(self, user=None, **kwargs):
        for f in self.finders:
            if user.can_use(f):
                return f
