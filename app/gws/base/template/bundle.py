import os
import gws
import gws.types as t
import gws.lib.mime
from . import core


class Config(gws.Config):
    templates: t.List[core.Config]
    defaults: t.Optional[t.List[core.Config]]
    withBuiltins: bool


class Props(gws.Props):
    items: t.List[core.Props]


class Object(gws.Node, gws.ITemplateBundle):
    def props_for(self, user):
        return Props(items=self.items)

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

    def find(self, subject: str = None, category: str = None, name: str = None, mime: str = None) -> t.Optional[gws.ITemplate]:
        items = self.items

        if mime:
            mime = gws.lib.mime.get(mime)
            items = [tpl for tpl in items if mime in tpl.mime_types]
        if subject:
            s = subject.lower()
            items = [tpl for tpl in items if s == tpl.subject]
        if category:
            s = category.lower()
            items = [tpl for tpl in items if s == tpl.category]
        if name:
            s = name.lower()
            items = [tpl for tpl in items if s == tpl.name]

        return items[0] if items else None


##

def create(root: gws.IRoot, cfg: gws.Config, parent: gws.Node = None, shared: bool = False) -> Object:
    return root.create_object(Object, cfg, parent, shared)


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
