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

    limit: int = 1000
    """search results limit"""


class Request(gws.Request):
    extent: t.Optional[gws.Extent]
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

        features = self._find(req, p)
        return Response(features=[gws.props(f, req.user, self) for f in features])

    def _find(self, req, p):

        project = req.require_project(p.projectUid)

        args = gws.SearchArgs(
            project=project,
            user=req.user,
            featureElements=['title', 'teaser', 'description'])

        if p.layerUids:
            args.layers = gws.compact(req.acquire(gws.ext.object.layer, uid) for uid in p.layerUids)

        if not args.layers:
            gws.log.debug(f'no layers found for {p!r}')
            return []

        if p.extent:
            args.bounds = gws.Bounds(crs=p.crs, extent=p.extent)

        args.limit = self.limit
        if p.limit:
            args.limit = min(int(p.limit), self.limit)

        if p.shapes:
            args.shapes = [gws.base.shape.from_props(s) for s in p.shapes]

        if p.tolerance:
            args.tolerance = gws.lib.uom.parse(p.tolerance, default=gws.Uom.PX)

        if p.resolution:
            args.resolution = p.resolution

        if p.keyword.strip():
            args.keyword = p.keyword.strip()

        return runner.run(req, args)
