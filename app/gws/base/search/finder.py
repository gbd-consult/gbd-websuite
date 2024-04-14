import gws
import gws.base.model
import gws.base.template
import gws.base.shape
import gws.config.util
import gws.lib.uom
import gws.types as t


class SpatialContext(t.Enum):
    map = 'map'
    """search in the map extent"""
    view = 'view'
    """search in the client view extent"""


class Config(gws.ConfigWithAccess):
    models: t.Optional[list[gws.ext.config.model]]
    """data models for features"""
    spatialContext: t.Optional[SpatialContext] = SpatialContext.map
    """spatial context for keyword searches"""
    templates: t.Optional[list[gws.ext.config.template]]
    """feature formatting templates"""
    title: t.Optional[str]
    """provider title"""
    withGeometry: bool = True
    """enable geometry search"""
    withKeyword: bool = True
    """enable keyword search"""
    withFilter: bool = True
    """enable filter search"""


class Object(gws.Finder):
    spatialContext: SpatialContext

    def configure(self):
        self.templates = []
        self.models = []

        self.withKeyword = self.cfg('withKeyword', default=True)
        self.withGeometry = self.cfg('withGeometry', default=True)
        self.withFilter = self.cfg('withFilter', default=True)

        self.spatialContext = self.cfg('spatialContext', default=SpatialContext.map)
        self.title = self.cfg('title', default='')

    ##

    def configure_models(self):
        return gws.config.util.configure_models(self)

    def configure_templates(self):
        return gws.config.util.configure_templates(self)

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

        if search.ogcFilter:
            if not self.supportsFilterSearch or not self.withFilter:
                gws.log.debug(f'can run: False: {self} {search.ogcFilter=} {self.supportsFilterSearch=} {self.withFilter=}')
                return False
            has_param = True

        return has_param

    def context_shape(self, search: gws.SearchQuery) -> gws.Shape:
        if search.shape:
            return search.shape
        if self.spatialContext == SpatialContext.view and search.bounds:
            return gws.base.shape.from_bounds(search.bounds)
        if search.project:
            return gws.base.shape.from_bounds(search.project.map.bounds)

    def run(self, search, user, layer=None):
        model = self.root.app.modelMgr.locate_model(self, layer, user=user, access=gws.Access.read)
        if not model:
            gws.log.debug(f'no model for {user.uid=} in finder {self.uid!r}')
            return []
        search = t.cast(gws.SearchQuery, gws.u.merge(search, shape=self.context_shape(search)))
        mc = gws.ModelContext(op=gws.ModelOperation.read, readMode=gws.ModelReadMode.search, user=user)
        return model.find_features(search, mc)
