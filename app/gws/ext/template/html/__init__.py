"""CX templates."""

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
    """HTML template"""
    pass


class ParsedTemplate(t.Data):
    page_size: t.Size
    map_size: t.Size
    margin: t.List[int]
    header: str
    footer: str
    text: str


class Object(gws.common.template.Object):
    map_placeholder = '__map__'

    @property
    def auto_uid(self):
        return None

    def configure(self):
        super().configure()

        self.parsed: ParsedTemplate = None
        self.parsed_time = 0
        self.path = self.var('path')
        self.text = self.var('text')

        uid = self.var('uid') or (misc.sha256(self.path) if self.path else self.klass.replace('.', '_'))
        self.set_uid(uid)

        self._parse()

    @property
    def page_size(self):
        self._check_cache()
        return self.parsed.page_size

    @property
    def map_size(self):
        self._check_cache()
        return self.parsed.map_size

    def render(self, context, render_output=None, out_path=None, format=None):
        self._check_cache()

        html = self._render_html(self.parsed.text, context)

        if format == 'pdf':
            if not out_path:
                raise ValueError('out_path required for pdf')

            out_path = gws.tools.pdf.render_html_with_map(
                html=html,
                map_render_output=render_output,
                map_placeholder=self.map_placeholder,
                page_size=self.parsed.page_size,
                margin=self.parsed.margin,
                out_path=out_path
            )

            return t.TemplateRenderOutput({
                'mimeType': gws.tools.mime.get('pdf'),
                'path': out_path
            })

        if out_path:
            with open(out_path, 'wt') as fp:
                fp.write(html)
            return t.TemplateRenderOutput({
                'mimeType': gws.tools.mime.get('html'),
                'path': out_path
            })

        return t.TemplateRenderOutput({
            'mimeType': gws.tools.mime.get('html'),
            'content': html
        })


    def add_headers_and_footers(self, context, in_path, out_path, format):
        if not self.parsed.header and not self.parsed.footer:
            return in_path

        text = self._frame_template()
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

    def _check_cache(self):
        if self.path and _file_mtime(self.path) > self.parsed_time:
            gws.log.debug(f'{self.path!r}: updated, reparsing')
            self._parse()

    def _parse(self):
        if self.path:
            with open(self.path, 'rt') as fp:
                text = fp.read()
        else:
            text = self.text

        self.parsed = ParsedTemplate({
            'page_size': [210, 297],
            'map_size': [100, 100],
            'margin': [10, 10, 10, 10],
            'header': '',
            'footer': '',
            'text': ''
        })

        # we cannot parse our html with bs4 or whatever, because it's a template,
        # and in a template, whitespace is critical, and a structural parser won't preserve it one-to-one

        tags_re = r'''(?xs)
            (
                <(?P<tag1> gws:page|gws:map) (?P<atts1> .*?) />
            )
            |
            (
                <(?P<tag2> gws:header|gws:footer) (?P<atts2> .*?)>
                    (?P<contents2> .*?)
                </(?P=tag2)>
            ) 
        '''

        self.parsed.text = re.sub(tags_re, lambda m: self._parse_tag(m.groupdict()), text)
        self.parsed_time = time.time()

    def _parse_tag(self, m):
        name = m.get('tag1') or m.get('tag2')
        contents = m.get('contents2') or ''

        a = m.get('atts1') or m.get('atts2') or ''
        atts = {}
        for k, v in re.findall(r'(\w+)="(.+?)"', a):
            atts[k] = v.strip()

        if name == 'gws:map':
            if 'width' in atts:
                self.parsed.map_size = _parse_size(atts)
            return f'''
                <div style="position:relative;width:{self.parsed.map_size[0]}mm; height:{self.parsed.map_size[1]}mm">
                    {self.map_placeholder}
                </div>
            '''.strip()

        # NB other tags don't return anything

        if name == 'gws:page':
            if 'width' in atts:
                self.parsed.page_size = _parse_size(atts)
            if 'margin' in atts:
                self.parsed.margin = _parse_margin(atts)

        if name == 'gws:header':
            self.parsed.header = contents.strip()

        if name == 'gws:footer':
            self.parsed.footer = contents.strip()

    def _frame_template(self):
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
                width:  {self.parsed.page_size[0]}mm;
                height: {self.parsed.page_size[1]}mm;
            }}
        '''

        body = f'''
            @each range(1, page_count + 1) as page:
                <table border=0 cellspacing=0 cellpadding=0>
                    <tr><td>{self.parsed.header}</td></tr>
                    <tr><td style="height:100%"></td></tr>
                    <tr><td>{self.parsed.footer}</td></tr>
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
            gws.log.warn(f'TEMPLATE: {e} at {path!r}:{line}')

        content = gws.tools.chartreux.render(
            text,
            context,
            silent=True,
            path=self.path or '<string>',
            error=err
        )

        return content


def _file_mtime(path):
    try:
        st = os.stat(path)
    except:
        return 1e20
    return int(st.st_mtime)


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
