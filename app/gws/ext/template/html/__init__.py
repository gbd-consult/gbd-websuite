"""CX templates."""

import re
import time
import os

import gws
import gws.common.template
import gws.gis.feature
import gws.gis.render
import gws.tools.mime
import gws.tools.os2
import gws.tools.pdf
import gws.tools.vendor.chartreux as chartreux

import gws.types as t


class Config(gws.common.template.Config):
    """HTML template"""
    pass


class ParsedTemplate(t.Data):
    page_size: t.Size
    map_size: t.Size
    margin: t.List[int]
    header: str
    footer: str
    text: str
    time: int


class Object(gws.common.template.Object):
    _parsed_template = None

    @property
    def parsed_template(self):
        if not self._parsed_template:
            self._parsed_template = self._parse()
        elif self.path and gws.tools.os2.file_mtime(self.path) > self._parsed_template.time:
            self._parsed_template = self._parse()
        return self._parsed_template

    @property
    def page_size(self):
        return self.parsed_template.page_size

    @property
    def map_size(self):
        return self.parsed_template.map_size

    def render(self, context, render_output=None, out_path=None, format=None):
        pt = self.parsed_template

        html = self._render_html(pt.text, context)

        if render_output:
            map_html = gws.gis.render.output_html(render_output)
            html = html.replace('[MAP_PLACEHOLDER]', map_html)

        if format == 'pdf':
            if not out_path:
                raise ValueError('out_path required for pdf')

            out_path = gws.tools.pdf.render_html(
                html,
                page_size=pt.page_size,
                margin=pt.margin,
                out_path=out_path
            )

            return t.TemplateOutput(mime=gws.tools.mime.get('pdf'), path=out_path)

        if out_path:
            gws.write_file(out_path, html)
            return t.TemplateOutput(mime=gws.tools.mime.get('html'), path=out_path)

        return t.TemplateOutput(mime=gws.tools.mime.get('html'), content=html)

    def add_headers_and_footers(self, context, in_path, out_path, format):
        pt = self.parsed_template

        if not pt.header and not pt.footer:
            return in_path

        text = self._frame_template(pt)
        html = self._render_html(text, context)

        if format == 'pdf':
            frame = gws.tools.pdf.render_html(
                html=html,
                page_size=self.page_size,
                margin=None,
                out_path=out_path + '-frame'
            )

            return gws.tools.pdf.merge(in_path, frame, out_path)

        return in_path

    def _parse(self):
        if self.path:
            with open(self.path, 'rt') as fp:
                text = fp.read()
        else:
            text = self.text

        pt = ParsedTemplate(
            page_size=[210, 297],
            map_size=[100, 100],
            margin=[10, 10, 10, 10],
            header='',
            footer='',
            text='',
            time=0
        )

        # we cannot parse our html with bs4 or whatever, because it's a template,
        # and in a template, whitespace is critical, and a structural parser won't preserve it one-to-one

        tags_re = r'''(?xs)
            (
                <(?P<tag1> gws:\w+) (?P<atts1> .*?) />
            )
            |
            (
                <(?P<tag2> gws:\w+) (?P<atts2> .*?)>
                    (?P<contents2> .*?)
                </(?P=tag2)>
            ) 
        '''

        pt.text = re.sub(tags_re, lambda m: self._parse_tag(pt, m.groupdict()), text)
        pt.time = time.time()

        return pt

    def _parse_tag(self, pt, m):
        name = m.get('tag1') or m.get('tag2')
        contents = (m.get('contents2') or '').strip()
        atts = _parse_atts(m.get('atts1') or m.get('atts2') or '')

        if name == 'gws:map':
            if 'width' in atts:
                pt.map_size = _parse_size(atts)
            return f'''
                <div style="position:relative;width:{pt.map_size[0]}mm; height:{pt.map_size[1]}mm">
                    [MAP_PLACEHOLDER]
                </div>
            '''.strip()

        # NB other tags don't return anything

        if name == 'gws:page':
            if 'width' in atts:
                pt.page_size = _parse_size(atts)
            if 'margin' in atts:
                pt.margin = _parse_margin(atts)

        if name == 'gws:header':
            pt.header = contents

        if name == 'gws:footer':
            pt.footer = contents

    def _frame_template(self, pt):
        css = f'''
            body, table, tr, td {{
                margin: 0;
                padding: 0
                border: none;
            }}
            table {{
                height: 100%;
                width: 100%;
            }}
            tr, td {{
                width: 100%
            }}
            body {{
                width:  {pt.page_size[0]}mm;
                height: {pt.page_size[1]}mm;
            }}
        '''

        body = f'''
            @each range(1, page_count + 1) as page:
                <table border=0 cellspacing=0 cellpadding=0>
                    <tr><td>{pt.header}</td></tr>
                    <tr><td style="height:100%"></td></tr>
                    <tr><td>{pt.footer}</td></tr>
                </table>
            @end
        '''

        return f'''
            <html>
                <style>{css}</style>
                <body>{body}</body>
            </html>
        '''

    def _render_html(self, text, context):

        context['gws'] = {
            'version': gws.VERSION,
            'endpoint': gws.SERVER_ENDPOINT,
        }

        def err(e, path, line):
            gws.log.warn(f'TEMPLATE: {e.__class__.__name__}:{e} in {path}:{line}')

        content = chartreux.render(
            text,
            context,
            path=self.path or '<string>',
            error=err
        )

        return content


def _parse_atts(a):
    return {k: v.strip() for k, v in re.findall(r'(\w+)="(.+?)"', a)}


def _parse_size(atts):
    return [
        int(atts['width']),
        int(atts['height']),
    ]


def _parse_margin(atts):
    ms = [int(x) for x in atts['margin'].split()]
    if len(ms) == 1:
        return [ms[0]] * 4
    if len(ms) == 2:
        return [ms[0], ms[1], ms[0], ms[1]]
    if len(ms) == 4:
        return ms
    raise ValueError('invalid margin spec')
