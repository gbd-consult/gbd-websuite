"""WFS provider."""

import gws
import gws.base.metadata
import gws.base.ows
import gws.lib.extent
import gws.lib.gis
import gws.lib.ows
import gws.lib.shape
import gws.types as t
from . import caps

"""
    References

    wfs 1.0.0: http://portal.opengeospatial.org/files/?artifact_id=7176 Sec 13.7.3
    wfs 1.1.0: http://portal.opengeospatial.org/files/?artifact_id=8339 Sec 14.7.3
    wfs 2.0.0: http://docs.opengeospatial.org/is/09-025r2/09-025r2.html Sec 11.1.3
    
    see also https://docs.geoserver.org/latest/en/user/services/wfs/basics.html
    
"""


class Config(gws.base.ows.provider.Config):
    pass


class Object(gws.base.ows.provider.Object):
    protocol = gws.OwsProtocol.WFS

    def configure(self):
        cc = caps.parse(self.get_capabilities())

        self.metadata = t.cast(gws.IMetaData, self.create_child(gws.base.metadata.Object, cc.metadata))
        self.version = cc.version
        self.operations = cc.operations
        self.source_layers = cc.source_layers
        self.supported_crs = cc.supported_crs

    def find_features(self, args):
        # first, find features within the bounds of given shapes,
        # then, filter features precisely
        # this is more performant than WFS spatial ops (at least for qgis)
        # and also works without spatial ops support on the provider side

        bounds = args.bounds
        shape = None

        if args.shapes:
            map_tolerance = 0

            if args.tolerance:
                n, u = args.tolerance
                map_tolerance = n * (args.resolution or 1) if u == 'px' else n

            shape = gws.lib.shape.union(args.shapes).tolerance_polygon(map_tolerance)
            bounds = shape.bounds

        our_crs = bounds.crs
        source_crs = self.source_crs or gws.lib.gis.best_crs(our_crs, self.supported_crs)
        bbox = gws.lib.extent.transform(bounds.extent, our_crs, source_crs)
        axis = gws.lib.gis.best_axis(source_crs, self.invert_axis_crs, gws.OwsProtocol.WFS, self.version)
        invert_axis = axis == 'yx'

        params = {}

        if invert_axis:
            bbox = gws.lib.gis.invert_bbox(bbox)
        params['BBOX'] = bbox

        if args.source_layer_names:
            params['TYPENAMES' if self.version >= '2.0.0' else 'TYPENAME'] = args.source_layer_names

        if args.limit:
            params['COUNT' if self.version >= '2.0.0' else 'MAXFEATURES'] = args.limit

        params['SRSNAME'] = source_crs
        params['VERSION'] = self.version

        params = gws.merge(params, args.get('params'))

        text = gws.lib.ows.request.get_text(**self.operation_args(gws.OwsVerb.GetFeature, params=params))
        features = gws.lib.ows.formats.read(text, crs=source_crs, invert_axis=invert_axis)

        if features is None:
            gws.log.error(f'WFS response not parsed, params={params!r}')
            return []

        if not shape:
            return features

        flt = []
        for f in features:
            if not f.shape:
                continue
            f.transform_to(our_crs)
            if f.shape.intersects(shape):
                flt.append(f)

        if len(flt) != len(features):
            gws.log.debug(f'WFS filter before={len(features)} after={len(flt)}')

        return flt


##

def create(root: gws.RootObject, cfg: gws.Config, shared: bool = False, parent: gws.Object = None) -> Object:
    if not shared:
        return t.cast(Object, root.create_object(Object, cfg, parent))
    return t.cast(Object, root.create_shared_object(Object, cfg, uid=gws.sha256(cfg)))
