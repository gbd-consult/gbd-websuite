"""Python format template."""

import re

import gws
import gws.common.template
import gws.gis.feature
import gws.gis.render
import gws.tools.mime
import gws.tools.misc
import gws.tools.os2
import gws.tools.pdf

import gws.types as t


class Config(gws.common.template.Config):
    """Format template"""
    pass


class Object(gws.common.template.Object):

    def render(self, context: dict, mro=None, out_path=None, legends=None, format=None):
        content = gws.tools.misc.format_placeholders(self.text, context)
        return t.TemplateOutput(mime=gws.tools.mime.get('text'), content=content)
