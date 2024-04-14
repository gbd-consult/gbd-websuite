"""Map-only template"""

import re

import gws
import gws.base.template
import gws.lib.htmlx
import gws.lib.mime
import gws.gis.render

gws.ext.new.template('map')


class Config(gws.base.template.Config):
    pass


class Props(gws.base.template.Props):
    pass


class Object(gws.base.template.Object):

    def render(self, tri):
        notify = tri.notify or (lambda *args: None)
        mp = tri.maps[0]

        mri = gws.MapRenderInput(
            backgroundColor=mp.backgroundColor,
            bbox=mp.bbox,
            center=mp.center,
            crs=tri.crs,
            dpi=tri.dpi,
            mapSize=self.pageSize,
            notify=notify,
            planes=mp.planes,
            project=tri.project,
            rotation=mp.rotation,
            scale=mp.scale,
            user=tri.user,
        )

        notify('begin_print')
        notify('begin_page')
        notify('begin_map')

        mro = gws.gis.render.render_map(mri)
        html = gws.gis.render.output_to_html_string(mro, wrap='fixed')

        notify('end_map')
        notify('end_page')
        notify('finalize_print')

        if not tri.mimeOut or tri.mimeOut == gws.lib.mime.HTML:
            notify('end_print')
            return gws.ContentResponse(mime=gws.lib.mime.HTML, content=html)

        if tri.mimeOut == gws.lib.mime.PDF:
            res_path = gws.u.printtemp('map.pdf')
            gws.lib.htmlx.render_to_pdf(
                html,
                out_path=res_path,
                page_size=self.pageSize,
            )
            notify('end_print')
            return gws.ContentResponse(contentPath=res_path)

        if tri.mimeOut == gws.lib.mime.PNG:
            res_path = gws.u.printtemp('map.png')
            gws.lib.htmlx.render_to_png(
                html,
                out_path=res_path,
                page_size=self.pageSize,
            )
            notify('end_print')
            return gws.ContentResponse(contentPath=res_path)

        raise gws.Error(f'invalid output mime: {tri.mimeOut!r}')
