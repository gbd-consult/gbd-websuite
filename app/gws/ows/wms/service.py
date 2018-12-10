import gws
import gws.types as t
import gws.gis.util
import gws.ows.error
import gws.ows.response
import gws.ows.request
import gws.tools.net
import gws.tools.xml3
from . import caps


class Service(gws.Object, t.ServiceInterface):
    def __init__(self):
        super().__init__()
        self.type = 'WMS'

    def configure(self):
        super().configure()

        self.url = self.var('url')

        if self.url:
            xml = gws.ows.request.get_text(
                self.url,
                service='WMS',
                request='GetCapabilities',
                params=self.var('params'),
                max_age=self.var('capsCacheMaxAge'))
        else:
            xml = self.var('xml')

        caps.parse(self, xml)

    """
        http://portal.opengeospatial.org/files/?artifact_id=1081&version=1&format=pdf Sec 7.2.2 and 7.3
        http://portal.opengeospatial.org/files/?artifact_id=14416 Sec 7.2.2 and 7.3
        
        see also https://docs.geoserver.org/latest/en/user/services/wms/basics.html
    
    """

    def info_format(self):
        formats = self.operations['GetFeatureInfo'].formats
        preferred = 'gml', 'text/xml', 'text/plain'

        for fmt in preferred:
            for f in formats:
                if fmt in f:
                    return f

        return formats[0]

    def find_features(self, args: t.FindFeaturesArgs):

        #  arbitrary width & height
        width = 1000
        height = 1000

        bbox = gws.gis.util.compute_bbox(
            args.point[0],
            args.point[1],
            args.crs,
            args.resolution,
            width,
            height
        )

        if args.axis == 'yx':
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]

        p = {
            'BBOX': bbox,
            'CRS' if self.version >= '1.3' else 'SRS': args.crs,
            'HEIGHT': height,
            'INFO_FORMAT': self.info_format(),
            'LAYERS': args.layers,
            'QUERY_LAYERS': args.layers,
            'STYLES': [''] * len(args.layers),
            'VERSION': self.version,
            'WIDTH': width,
        }

        if args.count:
            p['FEATURE_COUNT'] = args.count

        p['I' if self.version >= '1.3' else 'X'] = width >> 1
        p['J' if self.version >= '1.3' else 'Y'] = height >> 1

        p = gws.extend(p, args.params)

        url = self.operations['GetFeatureInfo'].get_url
        text = gws.ows.request.get_text(url, service='WMS', request='GetFeatureInfo', params=p)

        return gws.ows.response.parse(text, crs=args.crs)
