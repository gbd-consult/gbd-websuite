"""Model manager."""

from typing import Optional, cast

import gws
from . import default_model


class Object(gws.ModelManager):
    def get_model(self, uid, user=None, access=None):
        model = cast(gws.Model, self.root.get(uid, gws.ext.object.model))
        if not model:
            return
        if user and access and not user.can(access, model):
            return
        return model

    def find_model(self, *objects, user=None, access=None):
        for obj in objects:
            if not obj:
                continue
            p = self._find(obj, user, access)
            if p:
                return p

    def _find(self, obj, user, access):
        for model in getattr(obj, 'models', []):
            if user and access and not user.can(access, model):
                continue
            return model

    def editable_models(self, project, user):
        res = {}

        def _collect(obj):
            for model in getattr(obj, 'models', []):
                if model.isEditable and user.can_edit(model):
                    res[model.uid] = model

        if project.map:
            for la in project.map.rootLayer.descendants():
                _collect(la)

        _collect(project)
        _collect(self.root.app)

        queue = []
        for model in res.values():
            queue.extend(model.related_models())

        while queue:
            model = queue.pop(0)
            if model.uid in res:
                continue
            if not user.can_read(model):
                continue
            res[model.uid] = model
            queue.extend(model.related_models())

        return sorted(res.values(), key=lambda m: m.title)

    def default_model(self):
        return self.root.create_shared(default_model.Object, uid='gws.base.core.default_model.Object')
