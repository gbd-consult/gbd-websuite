"""Search API."""

import gws
import gws.base.action
import gws.base.template
import gws.base.feature
import gws.base.shape
import gws.lib.uom
import gws.types as t

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

        propses = self._get_features(req, p)
        return Response(features=propses)

    def _get_features(self, req: gws.IWebRequester, p: Request) -> list[gws.FeatureProps]:

        project = req.require_project(p.projectUid)
        search = gws.SearchQuery(project=project)

        if p.layerUids:
            search.layers = gws.compact(req.acquire(uid, gws.ext.object.layer) for uid in p.layerUids)

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
            search.tolerance = gws.lib.uom.parse(p.tolerance, gws.Uom.px)

        if p.resolution:
            search.resolution = p.resolution

        if p.keyword.strip():
            search.keyword = p.keyword.strip()

        results = self.root.app.searchMgr.run_search(search, req.user)
        if not results:
            return []

        gws.time_start(f'SEARCH.FIND: formatting')

        for res in results:
            res.feature.transform_to(search.bounds.crs)

        for res in results:
            templates = gws.compact(
                self.root.app.templateMgr.locate_template(res.finder, res.layer, project, user=req.user, subject=f'feature.{v}')
                for v in p.views or _DEFAULT_VIEWS
            )
            res.feature.render_views(templates, user=req.user, project=project, layer=res.layer)

        gws.time_end()

        propses = []
        mc = gws.ModelContext(op=gws.ModelOperation.read, readMode=gws.ModelReadMode.search, user=req.user)

        for res in results:
            p = res.feature.model.feature_to_view_props(res.feature, mc)
            propses.append(p)

        return propses
