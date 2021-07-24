"""WMS provder."""

import gws
import gws.base.metadata
import gws.base.ows
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
    capsLayersBottomUp: bool = False  #: layers are listed from bottom to top in the GetCapabilities document


class Object(gws.base.ows.provider.Object):
    service_type = 'WMS'

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.metadata = t.cast(gws.IMetaData, self.create_child(gws.base.metadata.Object, cc.metadata))
        self.service_version = cc.version
        self.operations = cc.operations
        self.source_layers = cc.source_layers
        self.supported_crs = cc.supported_crs

    def find_features(self, args):
        shape = args.shapes[0]
        if shape.type != gws.GeometryType.point:
            return []

        our_crs = shape.crs
        source_crs = self.source_crs or gws.lib.gis.best_crs(our_crs, self.supported_crs)
        shape = shape.transformed_to(source_crs)
        axis = gws.lib.gis.best_axis(source_crs, self.invert_axis_crs, 'WMS', self.service_version)

        #  draw a 1000x1000 bbox around a point
        width = 1000
        height = 1000

        bbox = gws.lib.gis.make_bbox(
            shape.x,
            shape.y,
            source_crs,
            args.resolution,
            width,
            height
        )

        invert_axis = axis == 'yx'
        if invert_axis:
            bbox = gws.lib.gis.invert_bbox(bbox)

        params = {
            'BBOX': bbox,
            'WIDTH': width,
            'HEIGHT': height,
            'CRS' if self.service_version >= '1.3' else 'SRS': source_crs,
            'INFO_FORMAT': self._info_format,
            'LAYERS': args.source_layer_names,
            'QUERY_LAYERS': args.source_layer_names,
            'STYLES': [''] * len(args.source_layer_names),
            'VERSION': self.service_version,
        }

        if args.limit:
            params['FEATURE_COUNT'] = args.limit

        params['I' if self.service_version >= '1.3' else 'X'] = width >> 1
        params['J' if self.service_version >= '1.3' else 'Y'] = height >> 1

        params = gws.merge(params, args.params)

        operation = self.operation('GetFeatureInfo')
        text = gws.lib.ows.request.get_text(operation.get_url, service='WMS', verb='GetFeatureInfo', params=params)
        features = gws.lib.ows.formats.read(text, crs=source_crs, invert_axis=invert_axis)

        if features is None:
            gws.log.error(f'WMS response not parsed, params={params!r}')
            return []

        return [f.transform_to(our_crs) for f in features]

    @gws.cached_property
    def _info_format(self):
        operation = self.operation('GetFeatureInfo')
        if not operation:
            return

        preferred = 'gml', 'text/xml', 'text/plain'

        for fmt in preferred:
            for f in operation.formats:
                if fmt in f:
                    return f

        return operation.formats[0]
