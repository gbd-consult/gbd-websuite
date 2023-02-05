"""CX templates"""

import re

import gws
import gws.base.template
import gws.lib.html2
import gws.lib.mime
import gws.lib.pdf
import gws.gis.render
import gws.lib.uom as units
import gws.lib.vendor.jump as jump
import gws.types as t

gws.ext.new.template('html')


class Config(gws.base.template.Config):
    pass

class Props(gws.base.template.Props):
    pass


_dummy_fn = lambda *args: None


class Object(gws.base.template.Object):

    def configure(self):
        if self.path:
            self.text = gws.read_file(self.path)
        self.load()

    def load(self):
        text = gws.read_file(self.path) if self.path else self.text

        parser = _Parser()
        jump.compile(
            text,
            path=self.path or '<string>',
            commands=parser,
        )

        self.page_size = parser.page_size
        self.map_size = parser.map_size

    def render(self, tri):
        notify = tri.notify or _dummy_fn

        notify('begin_print')

        if self.root.app.developer_option('template.always_reload'):
            if self.path:
                self.text = gws.read_file(self.path)

        parser = _Parser()
        rt = _Engine(self, tri, notify)

        html = self._do_render(self.text, self.prepare_args(tri.args), parser, rt)

        notify('finalize_print')

        mime = tri.mimeOut
        if not mime and self.mimes:
            mime = self.mimes[0]
        if not mime:
            mime = gws.lib.mime.HTML

        if mime == gws.lib.mime.HTML:
            notify('end_print')
            return gws.ContentResponse(mime=mime, content=html)

        if mime == gws.lib.mime.PDF:
            res_path = self._finalize_pdf(tri, html, parser)
            notify('end_print')
            return gws.ContentResponse(path=res_path)

        if mime == gws.lib.mime.PNG:
            res_path = self._finalize_png(tri, html, parser)
            notify('end_print')
            return gws.ContentResponse(path=res_path)

        raise gws.Error(f'invalid output mime: {tri.mimeOut!r}')

    def _do_render(self, text, args, parser, runtime):
        def err(e, path, line, env):
            gws.log.warn(f'TEMPLATE: {e.__class__.__name__}:{e} in {path}:{line}')

        if self.root.app.developer_option('template.raise_errors'):
            err = None

        if self.root.app.developer_option('template.save_compiled'):
            gws.write_file(
                gws.VAR_DIR + '/debug_template_' + gws.to_uid(self.path) + '_' + gws.sha256(text),
                jump.translate(
                    text,
                    commands=parser,
                    path=self.path or '<string>'))

        return jump.render(
            text,
            args,
            path=self.path or '<string>',
            error=err,
            runtime=runtime,
            commands=parser,
        )

    def _finalize_pdf(self, tri, html, parser):
        content_path = gws.tempname('final.pdf')
        has_frame = parser.header or parser.footer

        gws.lib.html2.render_to_pdf(
            html,
            out_path=content_path,
            page_size=parser.page_size,
            page_margin=parser.page_margin,
        )

        if not has_frame:
            return content_path

        args = gws.merge(tri.args, page_count=gws.lib.pdf.page_count(content_path))
        frame_text = self._page_frame_template(parser.header or '', parser.footer or '', parser.page_size)
        frame_html = self._do_render(frame_text, args, None, None)
        frame_path = gws.tempname('frame.pdf')
        gws.lib.html2.render_to_pdf(
            frame_html,
            out_path=frame_path,
            page_size=parser.page_size,
            page_margin=None,
        )

        comb_path = gws.tempname('comb.pdf')
        gws.lib.pdf.overlay(frame_path, content_path, comb_path)
        return comb_path

    def _finalize_png(self, tri, html, parser):
        res_path = gws.tempname('final.png')
        gws.lib.html2.render_to_png(
            html,
            out_path=res_path,
            page_size=parser.page_size,
            page_margin=parser.page_margin,
        )
        return res_path

    def _page_frame_template(self, header, footer, page_size):
        w, h, _ = page_size

        return f'''
            <html>
                <style>
                    body, table, tr, td {{
                        margin: 0;
                        padding: 0;
                        border: none;
                    }}
                    body, table {{
                        width:  {w}mm;
                        height: {h}mm;
                    }}
                    tr, td {{
                        width:  {w}mm;
                        height: {h // 2}mm;
                    }}
                </style>
                <body>
                    @each range(1, page_count + 1) as page:
                        <table border=0 cellspacing=0 cellpadding=0>
                            <tr valign="top"><td>{header}</td></tr>
                            <tr valign="bottom"><td>{footer}</td></tr>
                        </table>
                    @end
                </body>
            </html>
        '''


class _Parser:
    def __init__(self):
        self.page_size = None
        self.map_size = None
        self.page_margin = None
        self.header = None
        self.footer = None

    def command_page(self, cc, arg):
        cc.code.add(f'_RT.emit_begin_page()')
        ast = cc.expression.parse_args_ast(arg)
        self.page_size = self._parse_size(cc, ast)
        self.page_margin = self._parse_margin(cc, ast)
        args = cc.expression.walk_args(ast)
        cc.code.add('_PUSHBUF()')
        cc.parser.parse_until('end')
        cc.code.add(f'_RT.emit_end_page(_POPBUF(),{_comma(args)})')

    def command_map(self, cc, arg):
        ast = cc.expression.parse_args_ast(arg)
        if not self.map_size:
            # report the size of the first map only
            self.map_size = self._parse_size(cc, ast)
        args = cc.expression.walk_args(ast)
        cc.code.add(f'_RT.emit_map({_comma(args)})')

    def command_legend(self, cc, arg):
        ast = cc.expression.parse_args_ast(arg)
        args = cc.expression.walk_args(ast)
        cc.code.add(f'_RT.emit_legend({_comma(args)})')

    def command_header(self, cc, arg):
        self.header = cc.command.extract_text('').strip()

    def command_footer(self, cc, arg):
        self.footer = cc.command.extract_text('').strip()

    def _parse_size(self, cc, ast):
        w = self._parse_int(cc, ast, 'width')
        h = self._parse_int(cc, ast, 'height')
        return w, h, units.MM

    def _parse_int(self, cc, ast, name):
        val = self._get_arg(ast, name)
        if not val:
            return 0
        try:
            return int(cc.expression.constant(val))
        except:
            raise cc.error("invalid value for {name!r}")

    def _parse_margin(self, cc, ast):
        val = self._get_arg(ast, 'margin')
        if not val:
            return
        try:
            m = [int(x) for x in cc.expression.constant(val).split()]
            if len(m) == 1:
                return [m[0], m[0], m[0], m[0]]
            if len(m) == 2:
                return [m[0], m[1], m[0], m[1]]
            if len(m) == 4:
                return [m[0], m[1], m[2], m[3]]
        except:
            raise cc.error("invalid value for 'margin'")

    def _get_arg(self, ast, name):
        for kw in ast.keywords:
            if kw.arg == name:
                return kw.value


class _Engine(jump.Engine):
    def __init__(self, tpl: Object, tri: gws.TemplateRenderInput, notify: t.Callable):
        super().__init__()
        self.tpl = tpl
        self.tri = tri
        self.notify = notify or _dummy_fn
        self.map_count = 0
        self.page_count = 0
        self.legend_count = 0

    def emit_begin_page(self):
        self.notify('begin_page')
        if self.page_count > 0:
            self.prints('<div style="page-break-before: always"></div>')
        self.page_count += 1
        self.map_count = 0

    def emit_end_page(self, text, **kwargs):
        self.prints(text)
        self.notify('end_page')

    def emit_map(self, **kwargs):
        self.notify('begin_map')
        try:
            html = self._render_map(kwargs)
        except Exception as exc:
            gws.log.error(f'template {self.tpl.uid}: map error: {exc!r}')
            html = ''
        self.prints(html)
        self.notify('end_map')

    def _render_map(self, kwargs):
        index = int(kwargs.get('index')) if 'index' in kwargs else self.map_count
        self.map_count += 1
        mri = self._prepare_map(self.tri.maps[index], kwargs)
        mro = gws.gis.render.render_map(mri, self.notify)
        return gws.gis.render.output_to_html_string(mro)

    def _prepare_map(self, rim: gws.TemplateRenderInputMap, opts) -> gws.MapRenderInput:
        # these are mandatory
        width = int(opts['width'])
        height = int(opts['height'])
        return gws.MapRenderInput(
            background_color=rim.background_color,
            bbox=opts.get('bbox', rim.bbox),
            center=opts.get('center', rim.center),
            crs=self.tri.crs,
            dpi=self.tri.dpi,
            out_size=(width, height, units.MM),
            planes=rim.planes,
            rotation=opts.get('scale', rim.rotation),
            scale=opts.get('scale', rim.scale),
        )

    def emit_legend(self, **kwargs):
        try:
            html = self._render_legend(kwargs)
        except Exception as exc:
            gws.log.error(f'template {self.tpl.uid}: legend error: {exc!r}')
            html = ''
        self.prints(html)

    def _render_legend(self, kwargs):
        layers = self._legend_layers(kwargs)
        if not layers:
            raise gws.Error('no legend layers')

        lro = gws.gis.legend.render(gws.Legend(layers=layers))
        if not lro:
            raise gws.Error('no legend output')

        img_path = gws.gis.legend.to_image_path(lro)
        if not img_path:
            raise gws.Error('no legend image path')

        self.legend_count += 1
        return f'<img src="{img_path}"/>'

    def _legend_layers(self, kwargs):
        if 'layers' in kwargs:
            user = self.tri.user or self.tpl.root.app.auth.guestUser
            return gws.compact(user.acquire('gws.ext.layer', uid) for uid in kwargs['layers'].split())

        if not self.tri.maps:
            return

        if 'map' in kwargs:
            mp = self.tri.maps[int(kwargs['map'])]
            return mp.visible_layers

        return self.tri.maps[0].visible_layers


_comma = ','.join
