import gws
import gws.types as t


class Object(gws.Node, gws.IModelManager):
    def configure(self):
        self.models = []

        for p in self.var('models', default=[]):
            self.models.append(self.create_child(gws.ext.object.model, p))

    def add_model(self, m):
        self.models.append(m)

    def get_model_for(self, user=None, **kwargs):
        for m in self.models:
            if not user or user.can_use(m):
                return m
