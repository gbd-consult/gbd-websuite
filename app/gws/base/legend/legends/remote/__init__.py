import gws
import gws.lib.image
import gws.gis.ows
import gws.types as t

from ... import core

gws.ext.new.legend('remote')


class Config(core.Config):
    """External legend."""

    urls: t.List[gws.Url]
    """urls of externals legend images"""


class Object(core.Object):
    urls: t.List[str]

    def configure(self):
        self.urls = self.var('urls')

    def render(self, args=None):
        lros = []

        for url in self.urls:
            try:
                res = gws.gis.ows.request.get_url(url, max_age=self.cacheMaxAge)
                if not res.content_type.startswith('image/'):
                    raise gws.gis.ows.error.Error(f'wrong content type {res.content_type!r}')
                img = gws.lib.image.from_bytes(res.content)
                lro = gws.LegendRenderOutput(image=img, size=img.size())
                lros.append(lro)
            except gws.gis.ows.error.Error:
                gws.log.exception(f'render_legend: download failed url={url!r}')

        # NB even if there's only one image, it's not a bad idea to run it through the image converter
        return core.combine_outputs(gws.compact(lros), self.options)
