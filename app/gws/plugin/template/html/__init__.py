"""HTML templates.

The ``html`` template is written in the Jump template language. It produces
HTML and, for printing, PDF or PNG output. PDF and PNG are rendered from the
generated HTML (``gws.lib.htmlx``). The output type is the ``mimeOut`` of the
render input, or the first configured ``mimeTypes`` entry, or HTML. The
template source is given as ``text`` or as a file ``path``; a template file
is recompiled when it changes.

The arguments passed to a template can be accessed via the ``_ARGS`` object.

If a template explicitly returns a :obj:`gws.Response` object, the generated
text is ignored and the object is returned as the render result. Otherwise,
the result is a :obj:`gws.ContentResponse` object with the generated content.

This template supports the following extensions to Jump.

The ``@page`` command, which sets parameters for the printed page::

    @page (
        width="<page width in mm>"
        height="<page height in mm>"
        margin="<page margins in mm, one value or four values>"
    )

The ``@map`` command, which renders a map of the render input::

    @map (
        width="<width in mm>"
        height="<height in mm>"
        number="<optional, index of the map, 0 by default>"
        bbox="<optional, bounding box in projection units>"
        center="<optional, center coordinates in projection units>"
        scale="<optional, scale>"
        rotation="<optional, rotation in degrees>"
    )

The ``@legend`` command, which renders the legend of the visible layers of a
map, or of the given layers::

    @legend (
        number="<optional, index of the map, 0 by default>"
        layers="<optional, space separated list of layer UIDs>"
    )

The ``@header`` and ``@footer`` block commands, which define headers and
footers for multi-page printing::

    @header
        content
    @end header

    @footer
        content
    @end footer

Headers and footers are separate sub-templates, which receive the same
arguments as the main template and two additional arguments:

- ``numpages`` - the total number of pages in the document
- ``page`` - the current page number

They are rendered as a separate PDF, which is placed over the content PDF.

The ``@pagebreak`` command, which renders a page break.

Example::

    templates+ {
        subject "feature.title"
        type "html"
        text "{{name}}"
    }

    printers+ {
        template {
            type "html"
            path "/data/templates/print.cx.html"
            mimeTypes [ "application/pdf" ]
        }
    }
"""

from typing import Optional, cast

import gws
import gws.base.legend
import gws.base.template
import gws.gis.render
import gws.lib.htmlx
import gws.lib.mime
import gws.lib.osx
import gws.lib.pdf
import gws.lib.vendor.jump


@gws.ext.config.template('html')
class Config(gws.base.template.Config):
    """Jump template that renders HTML, PDF or image output."""

    path: Optional[gws.FilePath]
    """Template file."""
    text: str = ''
    """Template source."""


@gws.ext.props.template('html')
class Props(gws.base.template.Props):
    pass


@gws.ext.object.template('html')
class Object(gws.base.template.Object):
    """Jump template that renders HTML, PDF or PNG output."""

    path: str
    """Template file path."""
    text: str
    """Template source."""
    compiledTime: float = 0
    """Time of the last compilation."""
    compiledFn = None
    """Compiled template function."""

    def configure(self):
        self.path = self.cfg('path')
        self.text = self.cfg('text', default='')
        if not self.path and not self.text:
            raise gws.Error('either "path" or "text" required')

    def render(self, tri):
        self.notify(tri, 'begin_print')

        engine = Engine(self, tri)
        self.compile(engine)

        args = self.prepare_args(tri)
        res = engine.call(self.compiledFn, args=args, error=self.error_handler)

        if not isinstance(res, gws.Response):
            res = self.finalize(tri, res, args, engine)

        self.notify(tri, 'end_print')
        return res

    def compile(self, engine: 'Engine'):
        """Compile the template if needed.

        The template file is read again if it has changed since the last
        compilation. With the developer option ``template.always_reload``,
        the template is compiled on each call; with
        ``template.save_compiled``, the translated source is written to a
        debug file.

        Args:
            engine: Jump engine.
        """

        if self.path and (not self.text or gws.lib.osx.file_mtime(self.path) > self.compiledTime):
            self.text = gws.u.read_file(self.path)
            self.compiledFn = None

        if self.root.app.developer_option('template.always_reload'):
            self.compiledFn = None

        if not self.compiledFn:
            gws.log.debug(f'compiling {self} {self.path=}')
            if self.root.app.developer_option('template.save_compiled'):
                gws.u.write_debug_file(f'compiled_template_{self.uid}', engine.translate(self.text, path=self.path))

            self.compiledFn = engine.compile(self.text, path=self.path)
            self.compiledTime = gws.u.utime()

    def error_handler(self, exc, path, line, env):
        """Handle a template runtime error.

        The error is logged. With the developer option
        ``template.raise_errors``, the error is raised, otherwise rendering
        continues.

        Args:
            exc: The exception.
            path: Template path.
            line: Template line.
            env: Template environment.

        Returns:
            ``True`` to continue rendering, ``False`` to raise the error.
        """
        if self.root.app.developer_option('template.raise_errors'):
            gws.log.error(f'TEMPLATE_ERROR: {self}: {exc} IN {path}:{line}')
            return False

        gws.log.warning(f'TEMPLATE_ERROR: {self}: {exc} IN {path}:{line}')
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
        """Render a map of the render input as HTML.

        Args:
            tri: Template render input.
            width: Map width in mm.
            height: Map height in mm.
            index: Index of the map in ``tri.maps``.
            bbox: Bounding box, defaults to the bounding box of the map.
            center: Center, defaults to the center of the map.
            scale: Scale, defaults to the scale of the map.
            rotation: Rotation in degrees, defaults to the rotation of the map.

        Returns:
            HTML fragment.
        """
        self.notify(tri, 'begin_map')

        src: gws.MapRenderInput = tri.maps[index]
        dst: gws.MapRenderInput = gws.MapRenderInput(src)

        dst.bbox = bbox or src.bbox
        dst.center = center or src.center
        dst.targetCrs = tri.crs
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
        """Render a legend as an HTML image tag.

        Args:
            tri: Template render input.
            index: Index of the map in ``tri.maps``, whose visible layers are used by default.
            layers: Layer UIDs to use instead of the visible layers.

        Returns:
            An ``img`` tag pointing to the legend image, or ``None`` if there is no legend.
        """
        src: gws.MapRenderInput = tri.maps[index]

        layer_list = src.visibleLayers
        if layers:
            layer_list = gws.u.compact(tri.user.acquire(la) for la in gws.u.to_list(layers))

        if not layer_list:
            gws.log.debug(f'no layers for a legend')
            return

        legend = cast(gws.Legend, self.root.create_temporary(
            gws.ext.object.legend,
            type='combined',
            layerUids=[la.uid for la in layer_list]))

        lro = legend.render(tri.args)
        if not lro:
            gws.log.debug(f'empty legend render')
            return

        img_path = gws.u.ephemeral_path('legend.png')
        lro.image.to_path(img_path, gws.lib.mime.PNG)
        return f'<img src="{img_path}"/>'

    def render_page_break(self, tri: gws.TemplateRenderInput):
        """Render a page break.

        Args:
            tri: Template render input.

        Returns:
            HTML fragment.
        """
        self.notify(tri, 'page_break')
        return '<div style="page-break-after: always"></div>'

    ##

    def finalize(self, tri: gws.TemplateRenderInput, html: str, args: gws.TemplateArgs, main_engine: 'Engine'):
        """Convert the generated HTML into the output format.

        Args:
            tri: Template render input.
            html: Generated HTML.
            args: Template arguments.
            main_engine: Engine that rendered the HTML, holds the page settings, header and footer.

        Returns:
            Content response with HTML, or a PDF or PNG file.

        Raises:
            ``gws.Error``: If the output MIME type is not supported.
        """
        self.notify(tri, 'finalize_print')

        mime_type = tri.mimeOut
        if not mime_type and self.mimeTypes:
            mime_type = self.mimeTypes[0]
        if not mime_type:
            mime_type = gws.lib.mime.HTML

        if mime_type == gws.lib.mime.HTML:
            return gws.ContentResponse(mimeType=mime_type, content=html.lstrip())

        if mime_type == gws.lib.mime.PDF:
            res_path = self.finalize_pdf(tri, html, args, main_engine)
            return gws.ContentResponse(contentPath=res_path)

        if mime_type == gws.lib.mime.PNG:
            res_path = self.finalize_png(tri, html, args, main_engine)
            return gws.ContentResponse(contentPath=res_path)

        raise gws.Error(f'invalid output mime: {tri.mimeOut!r}')

    def finalize_pdf(self, tri: gws.TemplateRenderInput, html: str, args: gws.TemplateArgs, main_engine: 'Engine'):
        """Render the generated HTML as PDF.

        If a header or footer is defined, they are rendered on a separate
        PDF, one page per content page, which is placed over the content.

        Args:
            tri: Template render input.
            html: Generated HTML.
            args: Template arguments.
            main_engine: Engine that rendered the HTML.

        Returns:
            Path to the PDF file.
        """
        content_pdf_path = gws.u.ephemeral_path('content.pdf')

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

        args = gws.u.merge(args, numpages=gws.lib.pdf.page_count(content_pdf_path))

        frame_engine = Engine(self, tri)
        frame_text = self.frame_template(main_engine.header or '', main_engine.footer or '', page_size)
        frame_html = frame_engine.render(frame_text, args=args, error=self.error_handler)

        frame_pdf_path = gws.u.ephemeral_path('frame.pdf')

        gws.lib.htmlx.render_to_pdf(
            self.decorate_html(frame_html),
            out_path=frame_pdf_path,
            page_size=page_size,
            page_margin=None,
        )

        combined_pdf_path = gws.u.ephemeral_path('combined.pdf')
        gws.lib.pdf.overlay(frame_pdf_path, content_pdf_path, combined_pdf_path)

        return combined_pdf_path

    def finalize_png(self, tri: gws.TemplateRenderInput, html: str, args: gws.TemplateArgs, main_engine: 'Engine'):
        """Render the generated HTML as PNG.

        Args:
            tri: Template render input.
            html: Generated HTML.
            args: Template arguments.
            main_engine: Engine that rendered the HTML.

        Returns:
            Path to the PNG file.
        """
        out_png_path = gws.u.ephemeral_path('out.png')

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
        """Add the charset and, for file templates, a base URL to the HTML.

        The base URL is the template directory, so that relative paths in the
        template are resolved against it.

        Args:
            html: HTML text.

        Returns:
            Decorated HTML.
        """
        if self.path:
            d = gws.u.dirname(self.path)
            html = f'<base href="file://{d}/" />\n' + html
        html = '<meta charset="utf8" />\n' + html
        return html

    def frame_template(self, header, footer, page_size):
        """Create the Jump source of the header and footer frame.

        The frame has one page per content page, with the header at the top
        and the footer at the bottom.

        Args:
            header: Header template source.
            footer: Footer template source.
            page_size: Page size in mm.

        Returns:
            Template source.
        """
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
                        <table class="FRAME_TABLE" border="1" cellspacing="0" cellpadding="0">
                            <tr class="FRAME_TR" valign="top"><td class="FRAME_TD">{header}</td></tr>
                            <tr class="FRAME_TR" valign="bottom"><td class="FRAME_TD">{footer}</td></tr>
                        </table>
                    @end
                </body>
            </html>
        '''


##


class Engine(gws.lib.vendor.jump.Engine):
    """Jump engine with the commands of HTML templates.

    The engine collects the page settings, header and footer defined by the
    template while it renders.
    """

    pageMargin: Optional[gws.UomExtent] = None
    """Page margins set by ``@page``."""
    pageSize: gws.UomSize = []
    """Page size set by ``@page``."""
    header: str = ''
    """Header source set by ``@header``."""
    footer: str = ''
    """Footer source set by ``@footer``."""

    def __init__(self, template: Object, tri: Optional[gws.TemplateRenderInput] = None):
        """Create the engine.

        Args:
            template: The template being rendered.
            tri: Template render input. Without it, ``@map``, ``@legend`` and ``@pagebreak`` render nothing.
        """
        super().__init__()
        self.template = template
        self.tri = tri

    def def_page(self, **kw):
        """Handle the ``@page`` command.

        Args:
            **kw: Command arguments ``width``, ``height`` and ``margin``.
        """
        self.pageSize = (
            _scalar(kw, 'width', int, self.template.pageSize[0]),
            _scalar(kw, 'height', int, self.template.pageSize[1]),
            gws.Uom.mm)
        m = _list(kw, 'margin', int, 4)
        self.pageMargin = (*m, gws.Uom.mm) if m else self.template.pageMargin

    def def_map(self, **kw):
        """Handle the ``@map`` command.

        Args:
            **kw: Command arguments ``width``, ``height``, ``number``, ``bbox``, ``center``, ``scale`` and ``rotation``.

        Returns:
            HTML fragment.
        """
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
        """Handle the ``@legend`` command.

        Args:
            **kw: Command arguments ``number`` and ``layers``.

        Returns:
            HTML fragment.
        """
        if not self.tri:
            return
        return self.template.render_legend(
            self.tri,
            index=_scalar(kw, 'number', int, 0),
            layers=kw.get('layers'),
        )

    def def_pagebreak(self, **kw):
        """Handle the ``@pagebreak`` command.

        Args:
            **kw: Command arguments, not used.

        Returns:
            HTML fragment.
        """
        if not self.tri:
            return
        return self.template.render_page_break(self.tri)

    def mbox_header(self, text):
        """Handle the ``@header`` block command.

        Args:
            text: Header template source.
        """
        self.header = text

    def mbox_footer(self, text):
        """Handle the ``@footer`` block command.

        Args:
            text: Footer template source.
        """
        self.footer = text


def _scalar(kw, name, typ, default=None):
    """Read a scalar command argument and convert it to ``typ``."""
    val = kw.get(name)
    if val is None:
        return default
    return typ(val)


def _list(kw, name, typ, size, default=None):
    """Read a space separated list argument, a single value is repeated ``size`` times."""
    val = kw.get(name)
    if val is None:
        return default
    a = [typ(s) for s in val.split()]
    if len(a) == 1:
        return a * size
    if len(a) == size:
        return a
    raise TypeError('invalid length')
