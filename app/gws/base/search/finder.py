import gws
import gws.base.model
import gws.base.template
import gws.base.shape
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


class Object(gws.Node, gws.IFinder):
    spatialContext: SpatialContext
    title: str

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
        p = self.cfg('models')
        if p:
            self.models = gws.compact(self.configure_model(c) for c in p)
            return True

    def configure_model(self, cfg):
        return self.create_child(gws.ext.object.model, cfg)

    def configure_templates(self):
        p = self.cfg('templates')
        if p:
            self.templates = gws.compact(self.configure_template(cfg) for cfg in p)
            return True

    def configure_template(self, cfg):
        return self.create_child(gws.ext.object.template, cfg)

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

    def context_shape(self, search: gws.SearchQuery) -> gws.IShape:
        if search.shape:
            return search.shape
        if self.spatialContext == SpatialContext.view and search.bounds:
            return gws.base.shape.from_bounds(search.bounds)
        if search.project:
            return gws.base.shape.from_bounds(search.project.map.bounds)

    def run(self, search, user, layer=None):
        model = gws.base.model.locate(self.models, user, gws.Access.read)
        if not model and layer:
            model = gws.base.model.locate(layer.models, user, gws.Access.read)
        if not model:
            gws.log.debug(f'no model for {user.uid=} in finder {self.uid!r}')
            return []
        return model.find_features(
            t.cast(gws.SearchQuery, gws.merge(search, shape=self.context_shape(search))),
            user,
        )


##

def locate(
        finders: list[gws.IFinder],
        user: gws.IUser = None,
        uid: str = None
) -> t.Optional[gws.IFinder]:
    for finder in finders:
        if user and user.can_use(finder):
            continue
        if uid and finder.uid != uid:
            continue
        return finder
