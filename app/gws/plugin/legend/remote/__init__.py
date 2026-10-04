"""Remote legend.

Downloads legend images from URLs and combines them into one image.
Downloads are cached for ``cacheMaxAge``. Responses that are not images are
logged and skipped.

Example::

    legend {
        type "remote"
        urls ["https://example.com/legend_1.png", "https://example.com/legend_2.png"]
    }
"""

import gws
import gws.base.legend
import gws.base.ows.client
import gws.lib.image


@gws.ext.config.legend('remote')
class Config(gws.base.legend.Config):
    """Legend images loaded from external URLs."""

    urls: list[gws.Url]
    """URLs of external legend images."""


@gws.ext.object.legend('remote')
class Object(gws.base.legend.Object):
    """Remote legend."""

    urls: list[str]
    """URLs of the legend images."""

    def configure(self):
        self.urls = self.cfg('urls')

    def render(self, args=None):
        lro_list = []

        def _fetch(url):
            res = gws.base.ows.client.request.get_url(url)
            if not res.content_type.startswith('image/'):
                raise gws.ExternalServiceError(f'wrong content type {res.content_type!r}')
            return res.content

        for url in self.urls:
            try:
                content = gws.u.get_cached_object(
                    f'legend_{gws.u.sha256(url)}',
                    self.cacheMaxAge,
                    lambda: _fetch(url),
                )
                img = gws.lib.image.from_bytes(content)
                lro = gws.LegendRenderOutput(image=img, size=img.size())
                lro_list.append(lro)
            except gws.ExternalServiceError:
                gws.log.exception(f'render_legend: download failed url={url!r}')

        # NB even if there's only one image, it's not a bad idea to run it through the image converter
        return gws.base.legend.combine_outputs(lro_list, self.options)
