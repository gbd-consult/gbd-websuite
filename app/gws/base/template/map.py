"""Map-only template"""

import re

import gws
import gws.base.template
import gws.lib.legend
import gws.lib.html2
import gws.lib.os2
import gws.lib.mime
import gws.lib.pdf
import gws.lib.render
import gws.lib.units as units
import gws.types as t


@gws.ext.Config('template.map')
class Config(gws.base.template.Config):
    pass


@gws.ext.Object('template.map')
class Object(gws.base.template.Object):

    def render(self, tri, notify=None):
        if not tri.out_path:
            raise gws.Error(f'output path required')

        mp = tri.maps[0]

        mri = gws.MapRenderInput(
            background_color=mp.background_color,
            bbox=mp.bbox,
            center=mp.center,
            crs=tri.crs,
            dpi=tri.dpi,
            out_size=self.page_size,
            out_path=tri.out_path,
            planes=mp.planes,
            rotation=mp.rotation,
            scale=mp.scale,
        )

        notify = notify or (lambda a, b=None: None)

        notify('begin_print')
        notify('begin_page')
        notify('begin_map')

        mro = gws.lib.render.render_map(mri, notify)
        html = gws.lib.render.output_to_html_string(mro, wrap='fixed')

        notify('end_map')
        notify('end_page')
        notify('finalize_print')

        if not tri.out_mime or tri.out_mime == gws.lib.mime.HTML:
            notify('end_print')
            return gws.TemplateRenderOutput(mime=gws.lib.mime.HTML, content=html)

        if tri.out_mime == gws.lib.mime.PDF:
            gws.lib.html2.render_to_pdf(
                html,
                out_path=tri.out_path,
                page_size=self.page_size,
            )
            notify('end_print')
            return gws.TemplateRenderOutput(mime=gws.lib.mime.PDF, path=tri.out_path)

        if tri.out_mime == gws.lib.mime.PNG:
            gws.lib.html2.render_to_png(
                html,
                out_path=tri.out_path,
                page_size=self.page_size,
            )
            notify('end_print')
            return gws.TemplateRenderOutput(mime=gws.lib.mime.PDF, path=tri.out_path)

        raise gws.Error(f'invalid output mime: {tri.out_mime!r}')
