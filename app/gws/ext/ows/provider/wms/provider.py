"""WMS provder."""

"""
http://portal.opengeospatial.org/files/?artifact_id=1081&version=1&format=pdf Sec 7.2.2 and 7.3
http://portal.opengeospatial.org/files/?artifact_id=14416 Sec 7.2.2 and 7.3

see also https://docs.geoserver.org/latest/en/user/services/wms/basics.html
"""

import gws
import gws.common.ows.provider
import gws.gis.ows
import gws.gis.util
import gws.gis.util
import gws.tools.net
import gws.tools.xml3
import gws.types as t

from . import caps


class Object(gws.common.ows.provider.Object):
    def __init__(self):
        super().__init__()
        self.type = 'WFS'

    def configure(self):
        super().configure()

        if self.url:
            xml = gws.gis.ows.request.get_text(
                self.url,
                service='WMS',
                request='GetCapabilities',
                params=self.var('params'),
                max_age=self.var('capsCacheMaxAge'))
        else:
            xml = self.var('xml')

        caps.parse(self, xml)

    def find_features(self, args: t.SearchArgs) -> t.List[t.IFeature]:
        operation = self.operation('GetFeatureInfo')
        if not operation or not args.shapes:
            return []

        shape = args.shapes[0]
        if shape.type != t.GeometryType.point:
            return []

        our_crs = gws.gis.util.best_crs(shape.crs, self.supported_crs)
        shape = shape.transformed_to(our_crs)
        axis = gws.gis.util.best_axis(our_crs, self.invert_axis_crs, 'WMS', self.version)

        #  draw a 1000x1000 bbox around a point
        width = 1000
        height = 1000

        bbox = gws.gis.util.make_bbox(
            shape.x,
            shape.y,
            our_crs,
            args.resolution,
            width,
            height
        )

        invert_axis = axis == 'yx'
        if invert_axis:
            bbox = gws.gis.util.invert_bbox(bbox)

        p = {
            'BBOX': bbox,
            'WIDTH': width,
            'HEIGHT': height,
            'CRS' if self.version >= '1.3' else 'SRS': our_crs,
            'INFO_FORMAT': self._info_format,
            'LAYERS': args.source_layer_names,
            'QUERY_LAYERS': args.source_layer_names,
            'STYLES': [''] * len(args.source_layer_names),
            'VERSION': self.version,
        }

        if args.limit:
            p['FEATURE_COUNT'] = args.limit

        p['I' if self.version >= '1.3' else 'X'] = width >> 1
        p['J' if self.version >= '1.3' else 'Y'] = height >> 1

        p = gws.extend(p, args.params)

        text = gws.gis.ows.request.get_text(operation.get_url, service='WMS', request='GetFeatureInfo', params=p)
        found = gws.gis.ows.formats.read(text, crs=our_crs, invert_axis=invert_axis)

        if found is None:
            gws.p('WMS QUERY', p, 'NOT PARSED')
            return []

        gws.p('WMS QUERY', p, f'FOUND={len(found)}')
        return found

    @gws.cached_property
    def _info_format(self):
        op = self.operation('GetFeatureInfo')
        if not op:
            return
        preferred = 'gml', 'text/xml', 'text/plain'

        for fmt in preferred:
            for f in op.formats:
                if fmt in f:
                    return f

        return op.formats[0]
