import gws
import gws.types as t


class Object(gws.Node, gws.IModelManager):
    def configure(self):
        self.models = []

    def model(self, user=None, access=None, uid=None):
        for model in self.models:
            if user and access and not user.can(access, model, self.parent):
                continue
            if uid and model.uid != uid:
                continue
            return model

    def create_model(self, cfg):
        return self.add_model(self.create_child(gws.ext.object.model, cfg))

    def add_model(self, model):
        self.models.append(model)
        return self
