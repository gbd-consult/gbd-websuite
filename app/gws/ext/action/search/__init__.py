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


class Params(t.Params):
    bbox: t.Extent
    keyword: str = ''
    layerUids: t.List[str]
    pixelTolerance: int = 10
    limit: int = 0
    resolution: float
    shapes: t.Optional[t.List[t.ShapeProps]]
    withAttributes: bool = True
    withDescription: bool = True
    withGeometry: bool = True


class Object(gws.ActionObject):
    def __init__(self):
        super().__init__()
        self.limit: int = 0
        self.pixel_tolerance: int = 0
        self.feature_format: t.FormatObject = None

    def configure(self):
        super().configure()

        self.limit = self.var('limit')
        self.pixel_tolerance = self.var('pixelTolerance')
        self.feature_format = self.create_shared_object(
            'gws.common.format',
            'default_feature_description',
            gws.common.template.builtin_config('feature_format')
        )

    def api_find_features(self, req: gws.web.AuthRequest, p: Params) -> Response:
        """Perform a search"""

        project = req.require_project(p.projectUid)

        args = t.SearchArguments({
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

        results = gws.common.search.runner.run(req, args)
        ps = []

        for r in results:
            fmt = None
            if r.provider:
                fmt = r.provider.feature_format
            if not fmt and r.layer:
                fmt = r.layer.feature_format
            if not fmt and self.feature_format:
                fmt = self.feature_format

            if fmt:
                r.feature.apply_format(fmt)

            props = r.feature.props

            # security! raw attributes must not be exposed to the client
            delattr(props, 'attributes')

            ps.append(props)

        return Response({
            'features': ps,
        })
