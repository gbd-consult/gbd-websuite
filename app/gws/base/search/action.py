"""Search API."""

import gws
import gws.base.action
import gws.base.template
import gws.base.feature
import gws.base.shape
import gws.lib.uom
import gws.types as t

from . import runner

MAX_LIMIT = 1000
DEFAULT_TOLERANCE = 10, gws.Uom.PX
DEFAULT_VIEWS = ['title', 'teaser', 'description']


@gws.ext.config.action('search')
class Config(gws.base.action.Config):
    """Search action"""

    limit: int = 1000
    """search results limit"""
    tolerance: t.Optional[gws.Measurement]
    """default tolerance"""


class Request(gws.Request):
    extent: t.Optional[gws.Extent]
    crs: t.Optional[gws.CrsName]
    keyword: str = ''
    layerUids: t.List[str]
    limit: t.Optional[int]
    resolution: float
    shapes: t.Optional[t.List[gws.base.shape.Props]]
    tolerance: t.Optional[str]
    views: t.Optional[t.List[str]]


class Response(gws.Response):
    features: t.List[gws.base.feature.Props]


@gws.ext.object.action('search')
class Object(gws.base.action.Object):
    limit = 0
    tolerance: gws.Measurement

    def configure(self):
        self.limit = self.var('limit')
        self.tolerance = self.var('tolerance') or DEFAULT_TOLERANCE

    @gws.ext.command.api('searchFind')
    def find(self, req: gws.IWebRequester, p: Request) -> Response:
        """Perform a search"""

        features = self._find(req, p)
        return Response(features=[gws.props(f, req.user, self) for f in features])

    def _find(self, req, p):

        project = req.require_project(p.projectUid)

        search = gws.SearchArgs(
            project=project,
            user=req.user,
        )

        if p.layerUids:
            search.layers = gws.compact(req.acquire(uid, gws.ext.object.layer) for uid in p.layerUids)

        if not search.layers:
            gws.log.debug(f'no layers found for {p!r}')
            return []

        if p.extent:
            search.bounds = gws.Bounds(crs=p.crs or project.map.bounds.crs, extent=p.extent)
        else:
            search.bounds = project.map.bounds

        search.limit = self.limit
        if p.limit:
            search.limit = min(int(p.limit), self.limit)

        if p.shapes:
            shapes = [gws.base.shape.from_props(s) for s in p.shapes]
            search.shape = shapes[0] if len(shapes) == 1 else shapes[0].union(shapes[1:])

        search.tolerance = self.tolerance
        if p.tolerance:
            search.tolerance = gws.lib.uom.parse(p.tolerance, default=gws.Uom.PX)

        if p.resolution:
            search.resolution = p.resolution

        if p.keyword.strip():
            search.keyword = p.keyword.strip()

        results = runner.run(search, req.user)
        views = p.views or DEFAULT_VIEWS
        features = []

        for res in results:
            templates = []
            for v in views:
                tpl = gws.base.template.locate(res.finder.templates, user=req.user, subject=f'feature.{v}')
                if not tpl:
                    tpl = gws.base.template.locate(res.layer.templates, user=req.user, subject=f'feature.{v}')
                if tpl:
                    templates.append(tpl)

            res.feature.compute_values(gws.Access.read, req.user)
            res.feature.transform_to(search.bounds.crs)
            res.feature.render_views(templates, user=req.user, layer=res.layer)

            features.append(res.feature)

        return features
