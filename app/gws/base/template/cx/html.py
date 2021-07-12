"""CX templates"""

import re

import gws
import gws.types as t
import gws.base.template
import gws.lib.feature
import gws.lib.render
import gws.lib.mime
import gws.lib.os2
import gws.lib.pdf
import gws.lib.vendor.chartreux as chartreux


@gws.ext.Object('template.html')
class Object(gws.base.template.Object):
    header: str
    footer: str
    parsed_text: str

    def configure(self):
        self.page_size = 210, 297
        self.map_size = 100, 100
        self.margin = 10, 10, 10, 10
        self.header = ''
        self.footer = ''
        self.legend_use_all = False
        self.legend_mode = None

        self.legend_layer_uids = []
        self.parsed_text = ''

        self._load()

    def render(self, context: dict, args: gws.TemplateRenderArgs = None) -> gws.TemplateOutput:
        if self.root.application.developer_option('template.reparse'):
            self._load()

        def legend_func(layer_uid=None):
            if not args.legends:
                return ''
            if not layer_uid:
                return ''.join(args.legends.values())
            return args.legends.get(layer_uid, '')

        map_html = ''
        if args and args.mro:
            gws.lib.render.output_html(args.mro)
        context = gws.merge(context, GWS_MAP=map_html, GWS_LEGEND=legend_func)
        html = self._render_html(self.parsed_text, context)

        if args and args.format == 'pdf':
            if not args.out_path:
                raise ValueError('out_path required for pdf')
            gws.lib.pdf.render_html(
                html,
                page_size=self.page_size,
                margin=self.margin,
                out_path=args.out_path
            )
            return gws.TemplateOutput(mime=gws.lib.mime.PDF, path=args.out_path)

        if args and args.format == 'png':
            if not args.out_path:
                raise ValueError('out_path required for png')
            gws.lib.pdf.render_html_to_png(
                html,
                page_size=self.page_size,
                margin=self.margin,
                out_path=args.out_path
            )
            return gws.TemplateOutput(mime=gws.lib.mime.PNG, path=args.out_path)

        if args and args.out_path:
            gws.write_file(args.out_path, html)
            return gws.TemplateOutput(mime=gws.lib.mime.HTML, path=args.out_path)

        return gws.TemplateOutput(mime=gws.lib.mime.HTML, content=html)

    def add_headers_and_footers(self, context, in_path, out_path, format):
        if not self.header and not self.footer:
            return in_path

        text = self._frame_template()
        html = self._render_html(text, context)

        if format == 'pdf':
            frame = gws.lib.pdf.render_html(
                html=html,
                page_size=self.page_size,
                margin=None,
                out_path=out_path + '-frame'
            )
            return gws.lib.pdf.merge(in_path, frame, out_path)

        return in_path

    def _load(self):
        self.legend_layer_uids = []
        self.parsed_text = ''

        if self.path:
            self.text = gws.read_file(self.path)
        self._parse()

    def _parse(self):
        # we cannot parse our html with bs4 or whatever, because it's a template,
        # and in a template, whitespace is critical, and a structural parser won't preserve it one-to-one

        tags_re = r'''(?xs)
            (
                <(?P<tag1> gws:\w+) (?P<atts1> [^<>]*?) />
            )
            |
            (
                <(?P<tag2> gws:\w+) (?P<atts2> [^<>]*?)>
                    (?P<contents2> .*?)
                </(?P=tag2)>
            )
        '''

        self.legend_use_all = False

        self.parsed_text = re.sub(tags_re, lambda m: self._parse_tag(m.groupdict()), self.text)

        if self.legend_use_all:
            self.legend_layer_uids = []

    def _parse_tag(self, m):
        name = m.get('tag1') or m.get('tag2')
        contents = (m.get('contents2') or '').strip()
        atts = _parse_atts(m.get('atts1') or m.get('atts2') or '')

        if name == 'gws:map':
            if 'width' in atts:
                self.map_size = _parse_size(atts)
            return '{GWS_MAP|raw}'

        # NB other tags don't return anything

        if name == 'gws:page':
            if 'width' in atts:
                self.page_size = _parse_size(atts)
            if 'margin' in atts:
                self.margin = _parse_margin(atts)

        if name == 'gws:legend':
            self.legend_mode = gws.base.template.LegendMode.html
            if 'layer' in atts:
                html = ''
                for layer_uid in gws.as_list(atts['for']):
                    if layer_uid not in self.legend_layer_uids:
                        self.legend_layer_uids.append(layer_uid)
                    html += f'{{GWS_LEGEND({layer_uid!r})|raw}}'
                return html
            self.legend_use_all = True
            return '{GWS_LEGEND()|raw}'

        if name == 'gws:header':
            self.header = contents

        if name == 'gws:footer':
            self.footer = contents

    def _frame_template(self):
        css = f'''
            body, table, tr, td {{
                margin: 0;
                padding: 0;
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
                width:  {self.page_size[0]}mm;
                height: {self.page_size[1]}mm;
            }}
        '''

        body = f'''
            @each range(1, page_count + 1) as page:
                <table border=0 cellspacing=0 cellpadding=0>
                    <tr><td>{self.header}</td></tr>
                    <tr><td style="height:100%"></td></tr>
                    <tr><td>{self.footer}</td></tr>
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

        def err(e, path, line):
            gws.log.warn(f'TEMPLATE: {e.__class__.__name__}:{e} in {path}:{line}')

        if self.root.application.developer_option('template.save_compiled'):
            gws.write_file(
                gws.VAR_DIR + '/debug_template_' + gws.as_uid(self.path),
                chartreux.translate(
                    text,
                    path=self.path or '<string>'))

        content = chartreux.render(
            text,
            self.prepare_context(context),
            path=self.path or '<string>',
            error=err
        )

        return content


def _parse_atts(a):
    return {k: v.strip() for k, v in re.findall(r'(\w+)="(.+?)"', a)}


def _parse_size(atts):
    return int(atts['width']), int(atts['height'])


def _parse_margin(atts):
    m = [int(x) for x in atts['margin'].split()]
    if len(m) == 1:
        return gws.Extent(m[0], m[0], m[0], m[0])
    if len(m) == 2:
        return gws.Extent(m[0], m[1], m[0], m[1])
    if len(m) == 4:
        return gws.Extent(m[0], m[1], m[2], m[3])
    raise ValueError('invalid margin spec')
