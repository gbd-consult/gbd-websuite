"""WFS provider."""

import gws
import gws.lib.mime
import gws.lib.extent
import gws.lib.gis
import gws.lib.ows
import gws.lib.shape
import gws.types as t
from . import caps
from .. import core

"""
    References

    wfs 1.0.0: http://portal.opengeospatial.org/files/?artifact_id=7176 Sec 13.7.3
    wfs 1.1.0: http://portal.opengeospatial.org/files/?artifact_id=8339 Sec 14.7.3
    wfs 2.0.0: http://docs.opengeospatial.org/is/09-025r2/09-025r2.html Sec 11.1.3
    
    see also https://docs.geoserver.org/latest/en/user/services/wfs/reference.html
    
"""


class Config(core.ProviderConfig):
    pass


class Object(core.Provider):
    protocol = gws.OwsProtocol.WFS

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.metadata = cc.metadata
        self.source_layers = cc.source_layers
        self.supported_crs = cc.supported_crs
        self.version = cc.version
        self.operations.extend(cc.operations)

    def find_features(self, args: gws.SearchArgs) -> t.List[gws.IFeature]:
        # first, find features within the bounds of given shapes,
        # then, filter features precisely
        # this is more performant than WFS spatial ops (at least for qgis)
        # and also works without spatial ops support on the provider side

        bounds = args.bounds
        shape = None

        if args.shapes:
            map_tolerance = 0.0

            if args.tolerance:
                n, u = args.tolerance
                map_tolerance = n * (args.resolution or 1) if u == 'px' else n

            shape = gws.lib.shape.union(args.shapes).tolerance_polygon(map_tolerance)
            bounds = shape.bounds

        our_crs = bounds.crs
        request_crs = self.source_crs or gws.lib.gis.best_crs(bounds.crs, self.supported_crs)
        axis = gws.lib.gis.best_axis(request_crs, gws.OwsProtocol.WFS, self.version, self.invert_axis_crs)
        bbox = gws.lib.extent.transform(bounds.extent, bounds.crs, request_crs)
        if axis == gws.AXIS_YX:
            bbox = gws.lib.extent.swap_xy(bbox)

        params = {
            'BBOX': bbox,
            'SRSNAME': request_crs,
            'VERSION': self.version,
        }

        if args.source_layer_names:
            params['TYPENAMES' if self.version >= '2.0.0' else 'TYPENAME'] = args.source_layer_names

        if args.limit:
            params['COUNT' if self.version >= '2.0.0' else 'MAXFEATURES'] = args.limit

        params = gws.to_upper_dict(gws.merge(params, args.get('params')))

        fmt = self.preferred_formats.get(str(gws.OwsVerb.GetFeature))
        if fmt:
            params.setdefault('OUTPUTFORMAT', fmt)

        text = gws.lib.ows.request.get_text(**self.operation_args(gws.OwsVerb.GetFeature, params=params))
        features = gws.lib.ows.formats.read(text, crs=request_crs, axis=axis)

        if features is None:
            gws.log.debug(f'WFS NOT_PARSED params={params!r}')
            return []

        gws.log.debug(f'WFS FOUND={len(features)} params={params!r}')
        features = [f.transform_to(our_crs) for f in features]

        if not shape:
            return features

        filtered = []
        for f in features:
            if not f.shape:
                continue
            if f.shape.intersects(shape):
                filtered.append(f)

        if len(filtered) != len(features):
            gws.log.debug(f'WFS filter: before={len(features)} after={len(filtered)}')

        return filtered
