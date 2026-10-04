"""Base finder."""

from typing import Optional, cast

import gws
import gws.base.model
import gws.base.template
import gws.lib.shape
import gws.config.util
import gws.lib.uom


class SpatialContext(gws.Enum):
    """Area searched by keyword searches without a user geometry."""

    map = 'map'
    """Search the whole map extent."""
    view = 'view'
    """Search the extent currently visible in the client."""


class Config(gws.ConfigWithAccess):
    """Finder configuration."""

    models: Optional[list[gws.ext.config.model]]
    """Models used to read the features."""
    spatialContext: Optional[SpatialContext] = SpatialContext.map
    """Area searched by keyword searches without a user geometry."""
    templates: Optional[list[gws.ext.config.template]]
    """Templates to format the found features."""
    title: Optional[str]
    """Finder title."""
    category: Optional[str]
    """Category assigned to the results."""
    withGeometry: bool = True
    """Use this finder for geometry searches."""
    withKeyword: bool = True
    """Use this finder for keyword searches."""
    withFilter: bool = True
    """Use this finder for filter searches."""


class Object(gws.Finder):
    """Base class for finders.

    Reads the common finder options and decides whether a query can be run. Runs
    a query by reading the features through a model, within the area given by
    ``context_shape``. Subclasses declare which kinds of search they support and
    configure their models and templates, using ``configure_models`` and
    ``configure_templates``.
    """

    spatialContext: SpatialContext
    """Area searched by keyword searches without a user geometry."""

    def configure(self):
        self.templates = []
        self.models = []

        self.withKeyword = self.cfg('withKeyword', default=True)
        self.withGeometry = self.cfg('withGeometry', default=True)
        self.withFilter = self.cfg('withFilter', default=True)

        self.spatialContext = self.cfg('spatialContext', default=SpatialContext.map)
        self.title = self.cfg('title', default='')
        self.category = self.cfg('category', default='')

    ##

    def configure_models(self):
        """Create the models from the ``models`` config.

        Returns:
            ``True`` if models are configured.
        """
        return gws.config.util.configure_models_for(self)

    def configure_templates(self):
        """Create the templates from the ``templates`` config.

        Returns:
            ``True`` if templates are configured.
        """
        return gws.config.util.configure_templates_for(self)

    ##

    def can_run(self, search, user):
        has_param = False

        if search.keyword:
            if not self.supportsKeywordSearch or not self.withKeyword:
                gws.log.debug(f'can run: False: {self} {search.keyword=} {self.supportsKeywordSearch=} {self.withKeyword=}')
                return False
            has_param = True

        if search.shape:
            if not self.supportsGeometrySearch or not self.withGeometry:
                gws.log.debug(f'can run: False: {self} <shape> {self.supportsGeometrySearch=} {self.withGeometry=}')
                return False
            has_param = True

        if search.filter:
            if not self.supportsFilterSearch or not self.withFilter:
                gws.log.debug(f'can run: False: {self} {search.filter=} {self.supportsFilterSearch=} {self.withFilter=}')
                return False
            has_param = True

        return has_param

    def context_shape(self, search: gws.SearchQuery) -> gws.Shape:
        """Return the shape to search in.

        This is the query shape if given. Otherwise, it is the query bounds if
        ``spatialContext`` is ``view``, or the bounds of the project map.

        Args:
            search: Search query.

        Returns:
            The search shape, or ``None`` if there is none.
        """
        if search.shape:
            return search.shape
        if self.spatialContext == SpatialContext.view and search.bounds:
            return gws.lib.shape.from_bounds(search.bounds)
        if search.project:
            return gws.lib.shape.from_bounds(search.project.map.bounds)

    def run(self, search, user, layer=None):
        model = self.root.app.modelMgr.find_model(self, layer, user=user, access=gws.Access.read)
        if not model:
            gws.log.debug(f'no model for {user.uid=} in finder {self.uid!r}')
            return []
        search = cast(gws.SearchQuery, gws.u.merge(search, shape=self.context_shape(search)))
        mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.searchResults, user=user)
        return model.find_features(search, mc)
