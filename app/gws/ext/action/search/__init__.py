"""Search API."""

import gws
import gws.common.action
import gws.common.search.runner
import gws.common.template
import gws.gis.feature
import gws.gis.shape
import gws.tools.json2
import gws.tools.units
import gws.web.error

import gws.types as t

MAX_LIMIT = 1000


class Config(t.WithTypeAndAccess):
    """Search action"""

    limit: int = 1000  #: search results limit


class Response(t.Response):
    features: t.List[t.FeatureProps]


class Params(t.Params):
    bbox: t.Optional[t.Extent]
    crs: t.Optional[t.Crs]
    keyword: str = ''
    layerUids: t.List[str]
    limit: t.Optional[int]
    tolerance: t.Optional[str]
    resolution: float
    shapes: t.Optional[t.List[t.ShapeProps]]
    withAttributes: bool = True
    withDescription: bool = True
    withGeometry: bool = True


class Object(gws.common.action.Object):
    limit = 0

    def configure(self):
        super().configure()

        self.limit = self.var('limit')

    def api_find_features(self, req: t.IRequest, p: Params) -> Response:
        """Perform a search"""

        project = req.require_project(p.projectUid)

        bounds = t.Bounds(
            crs=p.crs or project.map.crs,
            extent=p.bbox or project.map.extent,
        )

        args = t.SearchArgs(
            bounds=bounds,
            keyword=(p.keyword or '').strip(),
            layers=gws.compact(req.acquire('gws.ext.layer', uid) for uid in p.layerUids),
            limit=min(p.limit, self.limit) if p.get('limit') else self.limit,
            project=project,
            resolution=p.resolution,
            shapes=[gws.gis.shape.from_props(s) for s in p.shapes] if p.get('shapes') else [],
            tolerance=(
                gws.tools.units.parse(p.tolerance, units=['px', 'm'], default='px')
                if p.get('tolerance') else None),
        )

        found = gws.common.search.runner.run(req, args)

        for f in found:
            f.transform_to(args.bounds.crs)
            f.apply_templates()
            f.apply_data_model()

        return Response(features=[f.props for f in found])
