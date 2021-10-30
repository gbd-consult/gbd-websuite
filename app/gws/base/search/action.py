"""Search API."""

import gws
import gws.base.api
import gws.base.template
import gws.lib.feature
import gws.lib.json2
import gws.lib.shape
import gws.lib.units
import gws.types as t
from . import runner

MAX_LIMIT = 1000


@gws.ext.Config('action.search')
class Config(gws.base.api.action.Config):
    """Search action"""

    limit: int = 1000  #: search results limit


class Response(gws.Response):
    features: t.List[gws.lib.feature.Props]


class Params(gws.Params):
    bbox: t.Optional[gws.Extent]
    crs: t.Optional[gws.Crs]
    keyword: str = ''
    layerUids: t.List[str]
    limit: t.Optional[int]
    tolerance: t.Optional[str]
    resolution: float
    shapes: t.Optional[t.List[gws.lib.shape.Props]]


@gws.ext.Object('action.search')
class Object(gws.base.api.action.Object):
    limit = 0

    def configure(self):
        self.limit = self.var('limit')

    @gws.ext.command('api.search.findFeatures')
    def find_features(self, req: gws.IWebRequest, p: Params) -> Response:
        """Perform a search"""

        project = req.require_project(p.projectUid)

        bounds = gws.Bounds(
            crs=p.crs or project.map.crs,
            extent=p.bbox or project.map.extent,
        )

        limit = self.limit
        if p.limit:
            limit = min(int(p.limit), self.limit)

        shapes = []
        if p.shapes:
            shapes = [gws.lib.shape.from_props(s) for s in p.shapes]

        tolerance = None
        if p.tolerance:
            tolerance = gws.lib.units.parse(p.tolerance, units=['px', 'm'], default='px')

        args = gws.SearchArgs(
            bounds=bounds,
            keyword=(p.keyword or '').strip(),
            layers=gws.compact(req.acquire('gws.ext.layer', uid) for uid in p.layerUids),
            limit=limit,
            project=project,
            resolution=p.resolution,
            shapes=shapes,
            tolerance=tolerance,
        )

        found = runner.run(req, args)

        for f in found:
            # @TODO only pull specified props from a feature
            f.transform_to(args.bounds.crs)
            f.apply_templates()
            f.apply_data_model()

        return Response(features=[gws.props(f, req.user, context=self) for f in found])
