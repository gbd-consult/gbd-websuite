"""WFS provider."""

import gws
import gws.gis.ows
import gws.base.shape
import gws.types as t

from . import caps
from .. import core, featureinfo

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
        self.version = cc.version
        self.operations.extend(cc.operations)

    def find_features(self, args: gws.SearchArgs, source_layers: t.List[gws.SourceLayer]) -> t.List[gws.IFeature]:
        # first, find features within the bounds of given shapes,
        # then, filter features precisely
        # this is more performant than WFS spatial ops (at least for qgis)
        # and also works without spatial ops support on the provider side

        bounds = args.bounds
        search_shape = None

        if args.shapes:
            geometry_tolerance = 0.0

            if args.tolerance:
                n, u = args.tolerance
                geometry_tolerance = n * (args.resolution or 1) if u == 'px' else n

            search_shape = gws.base.shape.union(args.shapes).tolerance_polygon(geometry_tolerance)
            bounds = search_shape.bounds

        ps = gws.gis.ows.client.prepared_search(
            inverted_crs=self.inverted_crs,
            limit=args.limit,
            bounds=bounds,
            protocol=self.protocol,
            protocol_version=self.version,
            request_crs=self.force_crs,
            request_crs_format=gws.CrsFormat.EPSG,
            source_layers=source_layers,
        )

        fmt = self.preferred_formats.get(gws.OwsVerb.GetFeature)
        if fmt:
            ps.params.setdefault('OUTPUTFORMAT', fmt)

        params = gws.merge(ps.params, args.params)

        text = gws.gis.ows.request.get_text(**self.request_args_for_operation(gws.OwsVerb.GetFeature, params=params))
        features = featureinfo.parse(text, crs=ps.request_crs, axis=ps.axis)

        if features is None:
            gws.log.debug(f'WFS NOT_PARSED params={params!r}')
            return []
        gws.log.debug(f'WFS FOUND={len(features)} params={params!r}')

        features = [f.transform_to(bounds.crs) for f in features]

        if not search_shape:
            return features

        filtered = []
        for f in features:
            if not f.shape:
                continue
            if f.shape.intersects(search_shape):
                filtered.append(f)

        if len(filtered) != len(features):
            gws.log.debug(f'WFS filter: before={len(features)} after={len(filtered)}')

        return filtered
