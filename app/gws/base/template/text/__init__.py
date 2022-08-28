"""Default text-only template."""

import gws
import gws.base.template
import gws.lib.vendor.jump
import gws.types as t


@gws.ext.config.template('text')
class Config(gws.base.template.Config):
    pass


@gws.ext.object.template('text')
class Object(gws.base.template.Object):
    """Text-only template"""

    def render(self, tri, notify=None):
        args = tri.args or {}

        args['gws'] = {
            'version': self.root.app.version,
            'endpoint': gws.SERVER_ENDPOINT,
        }

        def err(e, path, line, env):
            gws.log.warn(f'TEMPLATE: {e.__class__.__name__}:{e} in {path}:{line}')

        text = self.text
        if self.path:
            with open(self.path, 'rt') as fp:
                text = fp.read()

        content = gws.lib.vendor.jump.render(
            text,
            args,
            path=self.path or '<string>',
            error=err,
        )

        mime = self.mimes[0] if self.mimes else 'text/plain'
        return gws.ContentResponse(content=content, mime=mime)
