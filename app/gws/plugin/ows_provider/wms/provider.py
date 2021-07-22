"""WMS provder."""

import gws
import gws.base.metadata
import gws.base.ows.provider
import gws.lib.feature
import gws.lib.gis
import gws.lib.net
import gws.lib.ows
import gws.lib.xml2
import gws.types as t
from . import caps

"""
OGC documents:
    - OGC 01-068r3: WMS 1.1.1
    - OGC 06-042: WMS 1.3.0

see also https://docs.geoserver.org/latest/en/user/services/wms/basics.html

NB: layer order
our configuration lists layers top-to-bottom,
this also applies by default to WMS caps (like in qgis)

for servers with bottom-up caps, set capsLayersBottomUp=True 

the order of GetMap is always bottomUp:

> A WMS shall render the requested layers by drawing the leftmost in the list bottommost, 
> the next one over that, and so on.

OGC 06-042, 7.3.3.3 
"""


class Config(gws.base.ows.provider.Config):
    capsCacheMaxAge: gws.Duration = '1d'  #: max cache age for capabilities documents
    capsLayersBottomUp: bool = False  #: layers are listed from bottom to top in the GetCapabilities document
    getCapabilitiesParams: t.Optional[t.Dict]  #: additional parameters for GetCapabilities requests
    getMapParams: t.Optional[t.Dict]  #: additional parameters for GetMap requests


class Object(gws.base.ows.provider.Object):
    service_type = 'WMS'

    def configure(self):

        xml = gws.lib.ows.request.get_text(
            self.url,
            service='WMS',
            request='GetCapabilities',
            params=self.var('getCapabilitiesParams'),
            max_age=self.var('capsCacheMaxAge'))

        cc = caps.parse(xml)

        self.metadata = t.cast(gws.IMetaData, self.create_child(gws.base.metadata.Object, cc.metadata))
        self.service_version = cc.version
        self.operations = cc.operations
        self.source_layers = cc.source_layers
        self.supported_crs = cc.supported_crs

    def find_features(self, args: gws.SearchArgs) -> t.List[gws.lib.feature.Feature]:
        operation = self.operation('GetFeatureInfo')
        if not operation or not args.shapes:
            return []

        shape = args.shapes[0]
        if shape.type != gws.GeometryType.point:
            return []

        our_crs = gws.lib.gis.best_crs(shape.crs, self.supported_crs)
        shape = shape.transformed_to(our_crs)
        axis = gws.lib.gis.best_axis(our_crs, self.invert_axis_crs, 'WMS', self.service_version)

        #  draw a 1000x1000 bbox around a point
        width = 1000
        height = 1000

        bbox = gws.lib.gis.make_bbox(
            shape.x,
            shape.y,
            our_crs,
            args.resolution,
            width,
            height
        )

        invert_axis = axis == 'yx'
        if invert_axis:
            bbox = gws.lib.gis.invert_bbox(bbox)

        p = {
            'BBOX': bbox,
            'WIDTH': width,
            'HEIGHT': height,
            'CRS' if self.service_version >= '1.3' else 'SRS': our_crs,
            'INFO_FORMAT': self._info_format,
            'LAYERS': args.source_layer_names,
            'QUERY_LAYERS': args.source_layer_names,
            'STYLES': [''] * len(args.source_layer_names),
            'VERSION': self.service_version,
        }

        if args.limit:
            p['FEATURE_COUNT'] = args.limit

        p['I' if self.service_version >= '1.3' else 'X'] = width >> 1
        p['J' if self.service_version >= '1.3' else 'Y'] = height >> 1

        p = gws.merge(p, args.params)

        text = gws.lib.ows.request.get_text(operation.get_url, service='WMS', request='GetFeatureInfo', params=p)
        found = gws.lib.ows.formats.read(text, crs=our_crs, invert_axis=invert_axis)

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
