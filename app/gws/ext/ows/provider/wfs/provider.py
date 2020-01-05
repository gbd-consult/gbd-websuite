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
import gws.gis.extent
import gws.gis.ows
import gws.gis.shape
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
                service='WFS',
                request='GetCapabilities',
                params=self.var('params'),
                max_age=self.var('capsCacheMaxAge'))
        else:
            # @TODO offline caps not implemented yet
            xml = self.var('xml')

        caps.parse(self, xml)

    def find_features(self, args: t.SearchArgs) -> t.List[t.IFeature]:
        # first, find features within the bounds of given shapes,
        # then, filter features precisely
        # this is more performant than WFS spatial ops (at least for qgis)
        # and also works without spatial ops support on the provider side

        bounds = args.bounds
        shape = None
        if args.shapes:
            shape = gws.gis.shape.union(args.shapes)
            if shape.type == t.GeometryType.point:
                shape = shape.tolerance_buffer(args.get('tolerance'))
            bounds = shape.bounds

        our_crs = gws.gis.util.best_crs(bounds.crs, self.supported_crs)
        bbox = gws.gis.extent.transformed(bounds.extent, bounds.crs, our_crs)
        axis = gws.gis.util.best_axis(our_crs, self.invert_axis_crs, 'WFS', self.version)
        invert_axis = axis == 'yx'

        p = {}

        if invert_axis:
            bbox = gws.gis.util.invert_bbox(bbox)
        p['BBOX'] = bbox

        if args.source_layer_names:
            p['TYPENAMES' if self.version >= '2.0.0' else 'TYPENAME'] = args.source_layer_names

        if args.limit:
            p['COUNT' if self.version >= '2.0.0' else 'MAXFEATURES'] = args.limit

        p['SRSNAME'] = our_crs
        p['VERSION'] = self.version

        p = gws.extend(p, args.get('params'))

        url = self.operation('GetFeature').get_url
        text = gws.gis.ows.request.get_text(url, service='WFS', request='GetFeature', params=p)
        res = found = gws.gis.ows.formats.read(text, invert_axis=invert_axis)

        if found and shape:
            res = []
            for f in found:
                if not f.shape:
                    continue
                f.transform(shape.crs)
                if f.shape.intersects(shape):
                    res.append(f)

        if found is None:
            gws.p('WFS QUERY', p, 'NOT PARSED')
            return []

        gws.p('WFS QUERY', p, f'FOUND={len(found)} FILTERED={len(res)}')
        return res
