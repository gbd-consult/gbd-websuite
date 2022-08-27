import gws
import gws.lib.mime
import gws.types as t
from . import main

class Props(gws.Props):
    items: t.List[main.Props]


class Object(gws.Node, gws.ITemplateCollection):
    def configure(self):
        self.items = []

        for p in self.var('templates', default=[]):
            self.items.append(self.create_child(gws.ext.object.template, p))

        for p in self.var('defaults', default=[]):
            self.items.append(self.create_child(gws.ext.object.template, p))

    def props(self, user):
        return gws.Data(items=self.items)

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
