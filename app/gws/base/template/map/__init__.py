"""Map-only template"""

import re

import gws
import gws.base.template
import gws.lib.html2
import gws.lib.mime
import gws.gis.render


@gws.ext.config.template('map')
class Config(gws.base.template.Config):
    pass


@gws.ext.object.template('map')
class Object(gws.base.template.Object):

    def render(self, tri, notify=None):
        mp = tri.maps[0]

        mri = gws.MapRenderInput(
            background_color=mp.background_color,
            bbox=mp.bbox,
            center=mp.center,
            crs=tri.crs,
            dpi=tri.dpi,
            out_size=self.page_size,
            planes=mp.planes,
            rotation=mp.rotation,
            scale=mp.scale,
        )

        notify = notify or (lambda a, b=None: None)

        notify('begin_print')
        notify('begin_page')
        notify('begin_map')

        mro = gws.gis.render.render_map(mri, notify)
        html = gws.gis.render.output_to_html_string(mro, wrap='fixed')

        notify('end_map')
        notify('end_page')
        notify('finalize_print')

        if not tri.out_mime or tri.out_mime == gws.lib.mime.HTML:
            notify('end_print')
            return gws.ContentResponse(mime=gws.lib.mime.HTML, text=html)

        if tri.out_mime == gws.lib.mime.PDF:
            res_path = gws.tempname('map.pdf')
            gws.lib.html2.render_to_pdf(
                html,
                out_path=res_path,
                page_size=self.page_size,
            )
            notify('end_print')
            return gws.ContentResponse(path=res_path)

        if tri.out_mime == gws.lib.mime.PNG:
            res_path = gws.tempname('map.png')
            gws.lib.html2.render_to_png(
                html,
                out_path=res_path,
                page_size=self.page_size,
            )
            notify('end_print')
            return gws.ContentResponse(path=res_path)

        raise gws.Error(f'invalid output mime: {tri.out_mime!r}')
