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

    def get_template_for(self, user=None, **kwargs):
        for tpl in self.templates:
            if user and not user.can_use(tpl):
                continue
            p = kwargs.get('mime')
            if p and tpl.mimes and gws.lib.mime.get(p) not in tpl.mimes:
                continue
            p = kwargs.get('subject')
            if p and tpl.subject != p:
                continue
            return tpl

    def render_template(self, tri, user=None, **kwargs):
        tpl = self.get_template_for(user, **kwargs)
        if tpl:
            return tpl.render(tri, notify=kwargs.get('notify'))
