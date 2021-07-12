import gws
import gws.types as t
import gws.base.ows.provider
import gws.lib.ows
import gws.lib.xml2
from . import caps, types


class Object(gws.base.ows.provider.Object):
    def __init__(self):
        super().__init__()

        self.type = 'WMTS'
        self.url = ''

        self.source_layers: t.List[types.SourceLayer] = []
        self.matrix_sets: t.List[types.TileMatrixSet] = []

    def configure(self):
        

        self.url = self.var('url')

        if self.url:
            xml = gws.lib.ows.request.get_text(
                self.url,
                service='WMTS',
                request='GetCapabilities',
                params=self.var('params'),
                max_age=self.var('capsCacheMaxAge'))
        else:
            xml = self.var('xml')

        caps.parse(self, xml)
