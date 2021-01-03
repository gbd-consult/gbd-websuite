"""CX templates for Text-Only."""

import gws
import gws.common.template
import gws.tools.vendor.chartreux as chartreux

import gws.types as t


class Config(gws.common.template.Config):
    """text-only template"""
    pass


class Object(gws.common.template.Object):

    def render(self, context, format=None):
        def err(e, path, line):
            gws.log.warn(f'TEMPLATE: {e.__class__.__name__}:{e} in {path}:{line}')

        text = self.text
        if self.path:
            with open(self.path, 'rt') as fp:
                text = fp.read()

        content = chartreux.render(
            text,
            self.prepare_context(context),
            path=self.path or '<string>',
            error=err,
        )

        return t.Data({'content': content})
