"""CX templates for XML."""

import re
import time
import os

import gws
import gws.tools.mime
import gws.gis.feature
import gws.gis.render
import gws.tools.misc as misc
import gws.tools.pdf
import gws.types as t
import gws.tools.chartreux
import gws.common.template


class Config(t.TemplateConfig):
    """XML template"""
    pass


class Object(gws.common.template.Object):

    @property
    def auto_uid(self):
        return None

    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.text = self.var('text')

        uid = self.var('uid') or (misc.sha256(self.path) if self.path else self.klass.replace('.', '_'))
        self.set_uid(uid)

    def render(self, context, render_output=None, out_path=None, format=None):

        context['gws'] = {
            'version': gws.VERSION,
            'endpoint': gws.SERVER_ENDPOINT,
        }

        def err(e, path, line):
            gws.log.warn(f'TEMPLATE: {e} at {path!r}:{line}')

        text = self.text
        if self.path:
            with open(self.path, 'rt') as fp:
                text = fp.read()

        content = gws.tools.chartreux.render(
            text,
            context,
            path=self.path or '<string>',
            filter='xml',
            error=err,
        )

        return t.Data({'content': content})
