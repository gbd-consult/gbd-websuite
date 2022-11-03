import gws
import gws.lib.mime
import gws.types as t
from . import core


class Props(gws.Props):
    templates: t.List[core.Props]


class Object(gws.Node, gws.ITemplateManager):
    def configure(self):
        self.templates = []

        for p in self.var('templates', default=[]):
            self.templates.append(self.create_child(gws.ext.object.template, p))

        for p in self.var('defaults', default=[]):
            self.templates.append(self.root.create_shared(gws.ext.object.template, p))

    def props(self, user):
        return gws.Data(templates=self.templates)

    def find(self, subject=None, category=None, name=None, mime=None):
        templates = self.templates

        if mime:
            mime = gws.lib.mime.get(mime)
            templates = [tpl for tpl in templates if mime in tpl.mimes]
        if subject:
            s = subject.lower()
            templates = [tpl for tpl in templates if s == tpl.subject]
        if category:
            s = category.lower()
            templates = [tpl for tpl in templates if s == tpl.category]
        if name:
            s = name.lower()
            templates = [tpl for tpl in templates if s == tpl.name]

        return templates[0] if templates else None

    def render(self, tri, subject=None, category=None, name=None, mime=None, notify=None):
        tpl = self.find(subject, category, name, mime)
        if tpl:
            return tpl.render(tri, notify)
