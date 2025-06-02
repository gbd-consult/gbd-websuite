"""Remote legend."""

import gws
import gws.base.legend
import gws.base.ows.client
import gws.lib.image

gws.ext.new.legend('remote')


class Config(gws.base.legend.Config):
    """External legend."""

    urls: list[gws.Url]
    """Urls of external legend images."""


class Object(gws.base.legend.Object):
    urls: list[str]

    def configure(self):
        self.urls = self.cfg('urls')

    def render(self, args=None):
        lros = []

        for url in self.urls:
            try:
                res = gws.base.ows.client.request.get_url(url, max_age=self.cacheMaxAge)
                if not res.content_type.startswith('image/'):
                    raise gws.base.ows.client.Error(f'wrong content type {res.content_type!r}')
                img = gws.lib.image.from_bytes(res.content)
                lro = gws.LegendRenderOutput(image=img, size=img.size())
                lros.append(lro)
            except gws.base.ows.client.Error:
                gws.log.exception(f'render_legend: download failed url={url!r}')

        # NB even if there's only one image, it's not a bad idea to run it through the image converter
        return gws.base.legend.combine_outputs(gws.u.compact(lros), self.options)
