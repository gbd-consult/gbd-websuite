"""CX-based CSV templates."""

import io
import csv
import string

import gws
import gws.tools.mime
import gws.gis.feature
import gws.gis.render
import gws.tools.misc as misc
import gws.tools.pdf
import gws.types as t
import gws.tools.cx
import gws.common.template


class Config(t.TemplateConfig):
    """HTML template"""
    pass


class CsvCxTemplate(gws.tools.cx.Template):
    def __init__(self):
        self.rows = []

    def render(self, context, **opts):
        super().render(context)
        # NB ignore the render output

        buf = io.StringIO()
        fmt = opts.get('format', {})

        csv.writer(buf, **fmt).writerows(self.rows)
        return buf.getvalue()

    def command_row(self, compiler, arg):
        compiler.emit('self.rows.append([])')
        compiler.parse_until('end')

    def command_field(self, compiler, arg):
        compiler.emit('self.rows[-1].append(self.nows(%s))' % compiler.expression(arg))

    trans_ws = {ord(s): ' ' for s in string.whitespace}

    def nows(self, s):
        if isinstance(s, str):
            return s.translate(self.trans_ws).strip()
        return s


class Object(gws.common.template.Object):

    def configure(self):
        super().configure()
        self.page_size = [0, 0]
        self.map_size = [0, 0]
        self.path = self.var('path')
        self.text = self.var('text')

        # @TODO configurable
        self.format = {
            'quoting': csv.QUOTE_NONNUMERIC,
            'escapechar': '\\',
            'delimiter': ',',
            'doublequote': True,
            'lineterminator': '\r\n',
            'quotechar': '"',
        }

    def render(self, context, render_output=None, out_path=None, format=None):
        if self.path:
            with open(self.path, 'r') as fp:
                text = fp.read()
        else:
            text = self.text

        errors = []

        content = gws.tools.cx.render(
            text, context, errors,
            path=self.path or '<string>',
            base=CsvCxTemplate,
            format=self.format
        )

        for e in errors:
            gws.log.debug('TEMPLATE: ' + e)

        if out_path:
            with open(out_path, 'wt') as fp:
                fp.write(content)
            return t.TemplateRenderOutput({
                'mimeType': gws.tools.mime.get('csv'),
                'path': out_path
            })

        return t.TemplateRenderOutput({
            'mimeType': gws.tools.mime.get('csv'),
            'content': content
        })
