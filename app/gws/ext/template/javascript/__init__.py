"""Javascript template."""

import re

import gws
import gws.common.template
import gws.gis.feature
import gws.gis.render
import gws.tools.mime
import gws.tools.misc
import gws.tools.json2
import gws.tools.pdf

import gws.types as t


class Config(gws.common.template.Config):
    """Javascript template"""
    function: str
    arguments: t.Optional[t.List[t.Any]]


class Object(gws.common.template.Object):

    def render(self, context: dict, mro=None, out_path=None, legends=None, format=None):
        js = gws.tools.json2.to_string({
            'function': self.var('function'),
            'arguments': self.var('arguments', default=[])
        })
        content = 'javascript:' + js
        return t.TemplateOutput(mime=gws.tools.mime.get('text'), content=content)
