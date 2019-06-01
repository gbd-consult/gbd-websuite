import gws
import gws.web
import gws.gis.shape
import gws.gis.feature
import gws.tools.json2
import gws.common.template
import gws.common.search.runner

import gws.types as t

MAX_LIMIT = 1000


class Config(t.WithTypeAndAccess):
    """Search action"""

    limit: int = 1000  #: search results limit
    pixelTolerance: int = 5  #: pixel tolerance for geometry searches


class Response(t.Response):
    features: t.List[t.FeatureProps]


class Params(t.Data):
    bbox: t.Extent
    keyword: str = ''
    layerUids: t.List[str]
    pixelTolerance: int = 10
    limit: int = 0
    projectUid: str
    resolution: float
    shapes: t.Optional[t.List[t.ShapeProps]]
    withAttributes: bool = True
    withDescription: bool = True
    withGeometry: bool = True


class Object(gws.Object):
    def configure(self):
        super().configure()

        self.limit = self.var('limit')
        self.pixel_tolerance = self.var('pixelTolerance')
        self.feature_format = self.create_shared_object(
            'gws.common.format',
            'default_feature_description',
            gws.common.template.builtin_config('feature_format')
        )

    def api_find_features(self, req, p: Params) -> Response:
        """Perform a search"""

        project = req.require_project(p.projectUid)

        args = t.SearchArgs({
            'bbox': p.bbox,
            'crs': project.map.crs,
            'feature_format': self.feature_format,
            'keyword': (p.keyword or '').strip(),
            'layers': gws.compact(req.acquire('gws.ext.layer', uid) for uid in p.layerUids),
            'limit': min(p.limit, self.limit) if p.limit else self.limit,
            'project': project,
            'resolution': p.resolution,
            'shapes': [gws.gis.shape.from_props(s) for s in p.shapes] if p.get('shapes') else [],
            'tolerance': self.pixel_tolerance * p.resolution,
        })

        features = gws.common.search.runner.run(req, args)
        fprops = []

        for f in features:
            fmt = None
            if f.provider:
                fmt = f.provider.feature_format
            if not fmt and f.layer:
                fmt = f.layer.feature_format
            if not fmt and self.feature_format:
                fmt = self.feature_format

            if fmt:
                f.apply_format(fmt)

            p = f.props

            # security! raw attributes must not be exposed to the client
            delattr(p, 'attributes')

            fprops.append(p)

        return Response({
            'features': fprops,
        })
