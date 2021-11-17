import os

import gws
import gws.lib.mime
import gws.types as t

from . import core


class Props(gws.Props):
    items: t.List[core.Props]


class Object(gws.Node, gws.ITemplateBundle):
    def props_for(self, user):
        return gws.Data(items=self.items)

    def configure(self):
        self.items = []

        for cfg in self.var('items', default=[]):
            self.items.append(core.create(self.root, cfg, parent=self))

        # NB default templates are always shared
        for cfg in self.var('defaults', default=[]):
            self.items.append(core.create(self.root, cfg, parent=None, shared=True))

    def find(self, subject: str = None, category: str = None, name: str = None, mime: str = None) -> t.Optional[gws.ITemplate]:
        items = self.items

        if mime:
            mime = gws.lib.mime.get(mime)
            items = [tpl for tpl in items if mime in tpl.mimes]
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

def create(
        root: gws.IRoot,
        items: t.List[core.Config],
        parent: gws.INode,
        shared=False,
        defaults: t.Optional[t.List[core.Config]] = None,
) -> Object:
    cfg = gws.Data(
        items=items,
        defaults=defaults or [],
    )
    return root.create_object(Object, cfg, parent, shared)


##


_dir = os.path.dirname(__file__) + '/builtin_templates/'
_public = [{'role': 'all', 'type': 'allow'}]

_BUILTINS = [
]
