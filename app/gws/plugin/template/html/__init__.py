"""CX templates"""

import gws
import gws.base.legend
import gws.base.template
import gws.gis.render
import gws.lib.date
import gws.lib.htmlx
import gws.lib.mime
import gws.lib.osx
import gws.lib.pdf
import gws.lib.vendor.jump
import gws.types as t

gws.ext.new.template('html')


class Config(gws.base.template.Config):
    path: t.Optional[gws.FilePath]
    """path to a template file"""
    text: str = ''
    """template content"""


class Props(gws.base.template.Props):
    pass


class Object(gws.base.template.Object):
    path: str
    text: str
    compiledTime: float = 0
    compiledFn = None

    def configure(self):
        self.path = self.cfg('path')
        self.text = self.cfg('text', default='')
        if not self.path and not self.text:
            raise gws.Error('either "path" or "text" required')

    def render(self, tri):
        self.notify(tri, 'begin_print')

        engine = Engine(self, tri)
        self.compile(engine)

        args = self.prepare_args(tri.args)
        args['__renderUid'] = gws.random_string(8)
        html = engine.call(self.compiledFn, args=args, error=self.error_handler)

        res = self.finalize(tri, html, args, engine)

        self.notify(tri, 'end_print')
        return res

    def compile(self, engine: 'Engine'):

        if self.path and (not self.text or gws.lib.osx.file_mtime(self.path) > self.compiledTime):
            self.text = gws.read_file(self.path)
            self.compiledFn = None

        if self.root.app.developer_option('template.no_cache'):
            self.compiledFn = None

        if not self.compiledFn:
            gws.log.debug(f'compiling {self.uid!r} {self.path=}')
            if self.root.app.developer_option('template.save_compiled'):
                gws.write_file(
                    gws.ensure_dir(f'{gws.VAR_DIR}/debug') + f'/compiled_template_{self.uid}',
                    engine.translate(self.text, path=self.path)
                )

            self.compiledFn = engine.compile(self.text, path=self.path)
            self.compiledTime = gws.lib.date.utime()

    def error_handler(self, exc, path, line, env):
        rid = env.ARGS.get('__renderUid', '?')

        if self.root.app.developer_option('template.raise_errors'):
            gws.log.error(f'TEMPLATE_ERROR: {self.uid}/{rid}: {exc} IN {path}:{line}')
            for k, v in sorted(getattr(env, 'ARGS', {}).items()):
                gws.log.error(f'TEMPLATE_ERROR: {self.uid}/{rid}: ARGS {k}={v!r}')
            gws.log.error(f'TEMPLATE_ERROR: {self.uid}/{rid}: stop')
            return False

        gws.log.warning(f'TEMPLATE_ERROR: {self.uid}/{rid}: {exc} IN {path}:{line}')
        return True

    ##

    def render_map(
            self,
            tri: gws.TemplateRenderInput,
            width,
            height,
            index,
            bbox=None,
            center=None,
            scale=None,
            rotation=None,

    ):
        self.notify(tri, 'begin_map')

        src: gws.MapRenderInput = tri.maps[index]
        dst: gws.MapRenderInput = gws.MapRenderInput(src)

        dst.bbox = bbox or src.bbox
        dst.center = center or src.center
        dst.crs = tri.crs
        dst.dpi = tri.dpi
        dst.mapSize = width, height, gws.Uom.mm
        dst.rotation = rotation or src.rotation
        dst.scale = scale or src.scale
        dst.notify = tri.notify

        mro: gws.MapRenderOutput = gws.gis.render.render_map(dst)
        html = gws.gis.render.output_to_html_string(mro)

        self.notify(tri, 'end_map')
        return html

    def render_legend(
            self,
            tri: gws.TemplateRenderInput,
            index,
            layers,

    ):
        src: gws.MapRenderInput = tri.maps[index]

        layer_list = src.visibleLayers
        if layers:
            layer_list = gws.compact(tri.user.acquire(la) for la in gws.to_list(layers))

        if not layer_list:
            gws.log.debug(f'no layers for a legend')
            return

        legend = t.cast(gws.ILegend, self.root.create_temporary(
            gws.ext.object.legend,
            type='combined',
            layerUids=[la.uid for la in layer_list]))

        lro = legend.render(tri.args)
        if not lro:
            gws.log.debug(f'empty legend render')
            return

        img_path = gws.base.legend.output_to_image_path(lro)
        return f'<img src="{img_path}"/>'

    ##

    def finalize(self, tri: gws.TemplateRenderInput, html: str, args: dict, main_engine: 'Engine'):
        self.notify(tri, 'finalize_print')

        mime = tri.mimeOut
        if not mime and self.mimeTypes:
            mime = self.mimeTypes[0]
        if not mime:
            mime = gws.lib.mime.HTML

        if mime == gws.lib.mime.HTML:
            return gws.ContentResponse(mime=mime, content=html)

        if mime == gws.lib.mime.PDF:
            res_path = self.finalize_pdf(tri, html, args, main_engine)
            return gws.ContentResponse(path=res_path)

        if mime == gws.lib.mime.PNG:
            res_path = self.finalize_png(tri, html, args, main_engine)
            return gws.ContentResponse(path=res_path)

        raise gws.Error(f'invalid output mime: {tri.mimeOut!r}')

    def finalize_pdf(self, tri: gws.TemplateRenderInput, html: str, args: dict, main_engine: 'Engine'):
        content_pdf_path = gws.printtemp('content.pdf')

        page_size = main_engine.pageSize or self.pageSize
        page_margin = main_engine.pageMargin or self.pageMargin

        gws.lib.htmlx.render_to_pdf(
            self.decorate_html(html),
            out_path=content_pdf_path,
            page_size=page_size,
            page_margin=page_margin,
        )

        has_frame = main_engine.header or main_engine.footer
        if not has_frame:
            return content_pdf_path

        args = gws.merge(args, numpages=gws.lib.pdf.page_count(content_pdf_path))

        frame_engine = Engine(self, tri)
        frame_text = self.frame_template(main_engine.header or '', main_engine.footer or '', page_size)
        frame_html = frame_engine.render(frame_text, args=args, error=self.error_handler)

        frame_pdf_path = gws.printtemp('frame.pdf')

        gws.lib.htmlx.render_to_pdf(
            self.decorate_html(frame_html),
            out_path=frame_pdf_path,
            page_size=page_size,
            page_margin=None,
        )

        combined_pdf_path = gws.printtemp('combined.pdf')
        gws.lib.pdf.overlay(frame_pdf_path, content_pdf_path, combined_pdf_path)

        return combined_pdf_path

    def finalize_png(self, tri: gws.TemplateRenderInput, html: str, args: dict, main_engine: 'Engine'):
        out_png_path = gws.printtemp('out.png')

        page_size = main_engine.pageSize or self.pageSize
        page_margin = main_engine.pageMargin or self.pageMargin

        gws.lib.htmlx.render_to_png(
            self.decorate_html(html),
            out_path=out_png_path,
            page_size=page_size,
            page_margin=page_margin,
        )

        return out_png_path

    ##

    def decorate_html(self, html):
        if self.path:
            d = gws.dirname(self.path)
            html = f'<base href="file://{d}/" />\n' + html
        html = '<meta charset="utf8" />\n' + html
        return html

    def frame_template(self, header, footer, page_size):
        w, h, _ = page_size

        return f'''
            <html>
                <style>
                    body, .FRAME_TABLE, .FRAME_TR, .FRAME_TD {{ margin: 0; padding: 0; border: none; }}
                    body, .FRAME_TABLE {{ width:  {w}mm; height: {h}mm; }}
                    .FRAME_TR, .FRAME_TD {{ width:  {w}mm; height: {h // 2}mm; }}
                </style>
                <body>
                    @for page in range(1, numpages + 1)
                        <table class="FRAME_TABLE" border=0 cellspacing=0 cellpadding=0>
                            <tr class="FRAME_TR" valign="top"><td class="FRAME_TD">{header}</td></tr>
                            <tr class="FRAME_TR" valign="bottom"><td class="FRAME_TD">{footer}</td></tr>
                        </table>
                    @end
                </body>
            </html>
        '''


##


class Engine(gws.lib.vendor.jump.Engine):
    pageMargin: list[int] = []
    pageSize: gws.MSize = []
    header: str = ''
    footer: str = ''

    def __init__(self, template: Object, tri: t.Optional[gws.TemplateRenderInput] = None):
        super().__init__()
        self.template = template
        self.tri = tri

    def def_page(self, **kw):
        self.pageSize = (
            _scalar(kw, 'width', int, self.template.pageSize[0]),
            _scalar(kw, 'height', int, self.template.pageSize[1]),
            gws.Uom.mm)
        self.pageMargin = _list(kw, 'margin', int, 4, self.template.pageMargin)

    def def_map(self, **kw):
        if not self.tri:
            return
        return self.template.render_map(
            self.tri,
            width=_scalar(kw, 'width', int, self.template.mapSize[0]),
            height=_scalar(kw, 'height', int, self.template.mapSize[1]),
            index=_scalar(kw, 'number', int, 0),
            bbox=_list(kw, 'bbox', float, 4),
            center=_list(kw, 'center', float, 2),
            scale=_scalar(kw, 'scale', int),
            rotation=_scalar(kw, 'rotation', int),
        )

    def def_legend(self, **kw):
        if not self.tri:
            return
        return self.template.render_legend(
            self.tri,
            index=_scalar(kw, 'number', int, 0),
            layers=kw.get('layers'),
        )

    def mbox_header(self, text):
        self.header = text

    def mbox_footer(self, text):
        self.footer = text


def _scalar(kw, name, typ, default=None):
    val = kw.get(name)
    if val is None:
        return default
    return typ(val)


def _list(kw, name, typ, size, default=None):
    val = kw.get(name)
    if val is None:
        return default
    a = [typ(s) for s in val.split()]
    if len(a) == 1:
        return a * size
    if len(a) == size:
        return a
    raise TypeError('invalid length')
