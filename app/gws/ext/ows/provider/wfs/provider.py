"""WFS provider."""

"""
    References

    wfs 1.0.0: http://portal.opengeospatial.org/files/?artifact_id=7176 Sec 13.7.3
    wfs 1.1.0: http://portal.opengeospatial.org/files/?artifact_id=8339 Sec 14.7.3
    wfs 2.0.0: http://docs.opengeospatial.org/is/09-025r2/09-025r2.html Sec 11.1.3
    
    see also https://docs.geoserver.org/latest/en/user/services/wfs/basics.html
    
"""

import gws
import gws.common.ows.provider
import gws.tools.xml3
import gws.tools.net
import gws.gis.ows

import gws.types as t

from . import caps


class Object(gws.common.ows.provider.Object):
    def __init__(self):
        super().__init__()
        self.url = ''
        self.type = 'WFS'

    def configure(self):
        super().configure()

        self.url = self.var('url')

        if self.url:
            xml = gws.gis.ows.request.get_text(
                self.url,
                service='WFS',
                request='GetCapabilities',
                params=self.var('params'),
                max_age=self.var('capsCacheMaxAge'))
        else:
            xml = self.var('xml')

        caps.parse(self, xml)

    def find_features(self, args: t.SearchArguments) -> t.List[t.Feature]:

        p = {}
        invert_axis = args.get('axis') == 'yx'

        b = args.get('bbox')
        if b:
            if invert_axis:
                b = [b[1], b[0], b[3], b[2]]
            p['BBOX'] = b

        p['TYPENAMES' if self.version >= '2.0.0' else 'TYPENAME'] = args.layers

        if args.count:
            p['COUNT' if self.version >= '2.0.0' else 'MAXFEATURES'] = args.count

        p['SRSNAME'] = args.crs
        p['VERSION'] = self.version

        p = gws.extend(p, args.get('params'))

        url = self.operation('GetFeature').get_url
        text = gws.gis.ows.request.get_text(url, service='WFS', request='GetFeature', params=p)
        return gws.gis.ows.formats.read(text, invert_axis=invert_axis)
