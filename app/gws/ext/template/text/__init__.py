"""CX templates for Text-Only."""

import gws
import gws.base.template
import gws.lib.vendor.chartreux as chartreux

import gws.types as t


class Config(gws.base.template.Config):
    """text-only template"""
    pass


class Object(gws.base.template.Object):

    def render(self, context: dict, mro: t.MapRenderOutput = None, out_path: str = None, legends: dict = None, format: str = None) -> t.TemplateOutput:
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

        content = chartreux.render(
            text,
            context,
            path=self.path or '<string>',
            error=err,
        )

        mime = self.mime_types[0] if self.mime_types else 'text/plain'
        return t.TemplateOutput(content=content, mime=mime)
