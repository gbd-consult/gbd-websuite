"""CX templates for Text-Only."""

import gws
import gws.base.template
import gws.lib.vendor.chartreux as chartreux
import gws.types as t


@gws.ext.config.template('text')
class Config(gws.base.template.Config):
    pass


@gws.ext.object.template('text')
class Object(gws.base.template.Object):
    """Text-only template"""

    def render(self, tri, notify=None):
        context = tri.context or {}

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

        mime = self.mimes[0] if self.mimes else 'text/plain'
        return gws.ContentResponse(content=content, mime=mime)
