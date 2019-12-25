"""CX templates for Text-Only."""

import gws
import gws.common.template
import gws.tools.misc
import gws.tools.chartreux

import gws.types as t

class Config(t.TemplateConfig):
    """text-only template"""
    pass



class Object(gws.common.template.Object):

    @property
    def auto_uid(self):
        return None

    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.text = self.var('text')

        if self.path:
            fp = open(self.path, 'rt')
            fp.close()

        uid = self.var('uid') or (gws.tools.misc.sha256(self.path) if self.path else self.klass.replace('.', '_'))
        self.set_uid(uid)

    def render(self, context, render_output=None, out_path=None, format=None):
        context = context or {}

        context['gws'] = {
            'version': gws.VERSION,
            'endpoint': gws.SERVER_ENDPOINT,
        }

        def err(e, path, line):
            gws.log.warn(f'TEMPLATE: {e.__class__.__name__}:{e} in {path}:{line}')

        text = self.text
        if self.path:
            with open(self.path, 'rt') as fp:
                text = fp.read()

        content = gws.tools.chartreux.render(
            text,
            context,
            path=self.path or '<string>',
            error=err,
        )

        return t.Data({'content': content})
