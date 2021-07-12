"""CX templates for Text-Only."""

import gws
import gws.types as t
import gws.base.template
import gws.lib.vendor.chartreux as chartreux


@gws.ext.Object('template.text')
class Object(gws.base.template.Object):
    """Text-only template"""

    def render(self, context: dict, args: gws.TemplateRenderArgs = None) -> gws.TemplateOutput:
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
        return gws.TemplateOutput(content=content, mime=mime)
