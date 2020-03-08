import gws
import gws.common.ows.provider
import gws.tools.xml2
import gws.gis.ows

import gws.types as t

from . import caps, types


class Object(gws.common.ows.provider.Object):
    def __init__(self):
        super().__init__()

        self.type = 'WMTS'
        self.url = ''

        self.source_layers: t.List[types.SourceLayer] = []
        self.matrix_sets: t.List[types.TileMatrixSet] = []

    def configure(self):
        super().configure()

        self.url = self.var('url')

        if self.url:
            xml = gws.gis.ows.request.get_text(
                self.url,
                service='WMTS',
                request='GetCapabilities',
                params=self.var('params'),
                max_age=self.var('capsCacheMaxAge'))
        else:
            xml = self.var('xml')

        caps.parse(self, xml)
