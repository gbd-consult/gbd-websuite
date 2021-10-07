import os
import gws
import gws.types as t
from . import core


class Config(gws.Config):
    templates: t.List[core.Config]
    defaults: t.Optional[t.List[core.Config]]
    withBuiltins: bool


class Props:
    items: t.List[core.Props]


class Object(gws.Object, gws.ITemplateBundle):
    items: t.List[gws.ITemplate]

    @property
    def props(self):
        return gws.Props(items=self.items)

    def configure(self):
        self.items = []

        for cfg in self.var('templates', default=[]):
            self.items.append(core.create(self.root, cfg, parent=self))

        self._merge(self.var('defaults'), shared=False)
        if self.var('withBuiltins'):
            self._merge(_BUILTINS, shared=True)

    def _merge(self, cfgs, shared):
        if not cfgs:
            return
        for c in cfgs:
            if any(item.subject == c.get('subject') for item in self.items):
                continue
            self.items.append(core.create(
                self.root, c, parent=None if shared else self, shared=shared))

    def find(self, subject: str = None, category: str = None, mime: str = None) -> t.Optional[gws.ITemplate]:
        for tpl in self.items:
            ok = (
                    (not subject or subject == tpl.subject)
                    and (not category or category == tpl.category)
                    and (not mime or mime in tpl.mime_types))
            if ok:
                return tpl


##

def create(root: gws.RootObject, cfg: gws.Config, parent: gws.Object = None, shared: bool = False) -> Object:
    return t.cast(Object, root.create_object(Object, cfg, parent, shared))


##


_dir = os.path.dirname(__file__) + '/builtin_templates/'
_public = [{'role': 'all', 'type': 'allow'}]

_BUILTINS = [
    gws.Config(
        type='html',
        path=_dir + '/layer_description.cx.html',
        subject='layer.description',
        access=_public,
    ),
    gws.Config(
        type='html',
        path=_dir + '/project_description.cx.html',
        subject='project.description',
        access=_public,
    ),
    gws.Config(
        type='html',
        path=_dir + '/feature_description.cx.html',
        subject='feature.description',
        access=_public,
    ),
    gws.Config(
        type='html',
        path=_dir + '/feature_teaser.cx.html',
        subject='feature.teaser',
        access=_public,
    ),
]
