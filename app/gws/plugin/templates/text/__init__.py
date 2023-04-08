"""Default text-only template."""

import gws
import gws.base.template
import gws.lib.vendor.jump
import gws.lib.mime
import gws.types as t

gws.ext.new.template('text')


class Config(gws.base.template.Config):
    pass


class Props(gws.base.template.Props):
    pass


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

        out = gws.lib.vendor.jump.render(
            text,
            args,
            path=self.path or '<string>',
            error=err,
        )

        return gws.ContentResponse(
            mime=self.mimes[0] if self.mimes else gws.lib.mime.TXT,
            content=out)
