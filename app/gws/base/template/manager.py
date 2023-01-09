import gws
import gws.lib.mime
import gws.types as t
from . import core


class Props(gws.Props):
    templates: t.List[core.Props]


class Object(gws.Node, gws.ITemplateManager):
    def configure(self):
        self.templates = []

    def props(self, user):
        return gws.Data(templates=self.templates)

    def template(self, user=None, subject=None, mime=None):
        mt = gws.lib.mime.get(mime) if mime else None

        for tpl in self.templates:
            if user and not user.can_use(tpl):
                continue
            if mt and tpl.mimes and mt not in tpl.mimes:
                continue
            if subject and tpl.subject != subject:
                continue
            return tpl

    def create_template(self, cfg):
        return self.add_template(self.create_child(gws.ext.object.template, cfg))

    def add_template(self, tpl):
        self.templates.append(tpl)
        return tpl

    def render_template(self, tri,     user=None, subject=None, mime=None):
        tpl = self.template(user, subject, mime)
        if not tpl:
            return
        return tpl.render(tri)
