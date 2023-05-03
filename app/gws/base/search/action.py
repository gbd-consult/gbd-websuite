"""Search API."""

import gws
import gws.base.action
import gws.base.template
import gws.base.feature
import gws.base.shape
import gws.lib.uom
import gws.types as t

from . import runner

gws.ext.new.action('search')

_DEFAULT_VIEWS = ['title', 'teaser', 'description']
_DEFAULT_TOLERANCE = 10, gws.Uom.px


class Config(gws.base.action.Config):
    """Search action"""

    limit: int = 1000
    """search results limit"""
    tolerance: t.Optional[gws.Measurement]
    """default tolerance"""


class Props(gws.base.action.Props):
    pass


class Request(gws.Request):
    crs: t.Optional[gws.CrsName]
    extent: t.Optional[gws.Extent]
    keyword: str = ''
    layerUids: list[str]
    limit: t.Optional[int]
    resolution: float
    shapes: t.Optional[list[gws.base.shape.Props]]
    tolerance: t.Optional[str]
    views: t.Optional[list[str]]


class Response(gws.Response):
    features: list[gws.FeatureProps]


class Object(gws.base.action.Object):
    limit = 0
    tolerance: gws.Measurement

    def configure(self):
        self.limit = self.cfg('limit')
        self.tolerance = self.cfg('tolerance') or _DEFAULT_TOLERANCE

    @gws.ext.command.api('searchFind')
    def find(self, req: gws.IWebRequester, p: Request) -> Response:
        """Perform a search"""

        features = self._find(req, p)
        return Response(features=[gws.props(f, req.user, self) for f in features])

    def _find(self, req, p):

        project = req.require_project(p.projectUid)
        search = gws.SearchQuery(project=project)

        if p.layerUids:
            search.layers = gws.compact(req.acquire(uid, gws.ext.object.layer) for uid in p.layerUids)

        if not search.layers:
            gws.log.debug(f'no layers found for {p!r}')
            return []

        search.bounds = project.map.bounds
        if p.extent:
            search.bounds = gws.Bounds(crs=p.crs or project.map.bounds.crs, extent=p.extent)

        search.limit = self.limit
        if p.limit:
            search.limit = min(int(p.limit), self.limit)

        if p.shapes:
            shapes = [gws.base.shape.from_props(s) for s in p.shapes]
            search.shape = shapes[0] if len(shapes) == 1 else shapes[0].union(shapes[1:])

        search.tolerance = self.tolerance
        if p.tolerance:
            search.tolerance = gws.lib.uom.parse(p.tolerance, default=gws.Uom.px)

        if p.resolution:
            search.resolution = p.resolution

        if p.keyword.strip():
            search.keyword = p.keyword.strip()

        results = runner.run(search, req.user)
        views = p.views or _DEFAULT_VIEWS
        features = []

        for res in results:
            templates = []
            for v in views:
                tpl = gws.base.template.locate(res.finder, res.layer, user=req.user, subject=f'feature.{v}')
                if tpl:
                    templates.append(tpl)

            res.feature.compute_values(gws.Access.read, req.user)
            res.feature.transform_to(search.bounds.crs)
            res.feature.render_views(templates, user=req.user, layer=res.layer)

            features.append(res.feature)

        return features
