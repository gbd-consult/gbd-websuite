import os
import gws
import gws.types as t
from . import core


class Config(gws.Config):
    templates: t.List[core.Config]
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

        subjects = set(tpl.subject for tpl in self.items)

        if self.var('withBuiltins'):
            for cfg in _BUILTINS:
                if cfg.get('subject') not in subjects:
                    self.items.append(core.create(self.root, cfg, shared=True))

    def find(self, subject: str = None, category: str = None, mime: str = None) -> t.Optional[gws.ITemplate]:
        for tpl in self.items:
            ok = (
                    (not subject or subject == tpl.subject)
                    and (not category or category == tpl.category)
                    and (not mime or mime in tpl.mime_types))
            if ok:
                return tpl


##

def create(root: gws.RootObject, cfg: gws.Config, shared: bool = False, parent: gws.Object = None) -> Object:
    if not shared:
        return t.cast(Object, root.create_object(Object, cfg, parent))

    uids = []
    for c in cfg.get('templates', default=[]):
        uids.append(c.get('uid') or c.get('path') or gws.sha256(c.get('text')))
    return t.cast(Object, root.create_shared_object(Object, cfg, gws.sha256(sorted(uids))))


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
