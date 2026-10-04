"""Search action."""

from typing import Optional

import gws
import gws.base.model
import gws.base.action
import gws.base.template
import gws.base.feature
import gws.base.shape
import gws.lib.uom


_DEFAULT_VIEWS = ['title', 'teaser', 'description']
_DEFAULT_TOLERANCE = 10, gws.Uom.px


@gws.ext.config.action('search')
class Config(gws.base.action.Config):
    """Runs searches in the project finders and returns the found features."""

    limit: int = 1000
    """Max. number of search results."""
    tolerance: Optional[gws.UomValueStr]
    """Default tolerance for geometry searches."""
    categories: Optional[list[str]]
    """Result categories users can filter search results by."""


@gws.ext.props.action('search')
class Props(gws.base.action.Props):
    categories: list[str]


class Request(gws.Request):
    """Request of the ``searchFind`` command."""

    crs: Optional[gws.CrsName]
    """CRS of the extent, defaults to the map CRS."""
    extent: Optional[gws.Extent]
    """Search extent, defaults to the map extent."""
    keyword: str = ''
    """Keyword to search for."""
    layerUids: list[str]
    """Layers whose finders are used in addition to the project and application finders."""
    limit: Optional[int]
    """Max. number of results, cannot exceed the configured limit."""
    resolution: float
    """Pixel resolution for geometry searches."""
    shapes: Optional[list[gws.base.shape.Props]]
    """Shapes to search in; several shapes are combined into one."""
    tolerance: Optional[str]
    """Tolerance for geometry searches, a value with a unit, in pixels if no unit is given."""
    views: Optional[list[str]]
    """Feature views to render, defaults to ``title``, ``teaser`` and ``description``."""
    categories: Optional[list[str]]
    """Result categories."""
    withCategories: bool
    """Return result categories."""


class Response(gws.Response):
    """Response of the ``searchFind`` command."""

    features: list[gws.FeatureProps]
    """Found features with rendered views."""


@gws.ext.object.action('search')
class Object(gws.base.action.Object):
    """Search action, runs searches for the client."""

    limit = 0
    """Max. number of results, 0 for no limit."""
    tolerance: gws.UomValue
    """Default tolerance for geometry searches."""
    categories: list[str] = []
    """Result categories users can filter by."""

    def configure(self):
        self.limit = self.cfg('limit') or 0
        self.tolerance = self.cfg('tolerance') or _DEFAULT_TOLERANCE
        self.categories = self.cfg('categories') or []

    def props(self, user):
        return gws.u.merge(super().props(user), categories=self.categories)

    @gws.ext.command.api('searchFind')
    def find(self, req: gws.WebRequester, p: Request) -> Response:
        """Run a search and return the found features."""

        return Response(features=self._get_features(req, p))

    def _get_features(self, req: gws.WebRequester, p: Request) -> list[gws.FeatureProps]:
        project = req.user.require_project(p.projectUid)
        search = gws.SearchQuery(project=project)

        if p.layerUids:
            search.layers = gws.u.compact(req.user.acquire(uid, gws.ext.object.layer) for uid in p.layerUids)

        search.bounds = project.map.bounds
        if p.extent:
            search.bounds = gws.Bounds(crs=p.crs or project.map.bounds.crs, extent=p.extent)

        search.limit = self.limit
        if p.limit and int(p.limit) > 0:
            search.limit = min(int(p.limit), self.limit) if self.limit else int(p.limit)

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

        mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.searchResults, project=project, user=req.user)

        for res in results:
            templates = gws.u.compact(
                self.root.app.templateMgr.find_template(f'feature.{v}', where=[res.finder, res.layer, project], user=req.user)
                for v in p.views or _DEFAULT_VIEWS
            )
            res.feature.render_views(templates, user=req.user, project=project, layer=res.layer)

        return [res.feature.model.feature_to_view_props(res.feature, mc) for res in results]
