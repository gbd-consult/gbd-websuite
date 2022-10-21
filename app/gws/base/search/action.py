"""Search API."""

import gws
import gws.base.action
import gws.base.feature
import gws.base.shape
import gws.lib.uom
import gws.types as t

from . import runner

MAX_LIMIT = 1000


@gws.ext.config.action('search')
class Config(gws.base.action.Config):
    """Search action"""

    limit: int = 1000  #: search results limit


class Request(gws.Request):
    bbox: t.Optional[gws.Extent]
    crs: t.Optional[gws.CrsName]
    keyword: str = ''
    layerUids: t.List[str]
    limit: t.Optional[int]
    resolution: float
    shapes: t.Optional[t.List[gws.base.shape.Props]]
    tolerance: t.Optional[str]


class Response(gws.Response):
    features: t.List[gws.base.feature.Props]


@gws.ext.object.action('search')
class Object(gws.base.action.Object):
    limit = 0

    def configure(self):
        self.limit = self.var('limit')

    @gws.ext.command.api('searchFind')
    def find(self, req: gws.IWebRequester, p: Request) -> Response:
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
            shapes = [gws.base.shape.from_props(s) for s in p.shapes]

        tolerance = None
        if p.tolerance:
            tolerance = gws.lib.uom.parse(p.tolerance, default=gws.Uom.PX)

        layers = []
        if p.layerUids:
            layers = gws.compact(req.acquire(gws.ext.object.layer, uid) for uid in p.layerUids)

        args = gws.SearchArgs(
            bounds=bounds,
            keyword=(p.keyword or '').strip(),
            layers=layers,
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
            # f.apply_data_model()

        return Response(features=[gws.props(f, req.user, context=self) for f in found])
