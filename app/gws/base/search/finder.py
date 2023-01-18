import gws
import gws.base.model
import gws.base.template
import gws.base.shape
import gws.lib.uom
import gws.types as t

_DEFAULT_TOLERANCE = 10, gws.Uom.px


class SpatialContext(t.Enum):
    MAP = 'map'
    """search in the map extent"""
    VIEW = 'view'
    """search in the client view extent"""


class Config(gws.ConfigWithAccess):
    models: t.Optional[t.List[gws.ext.config.model]]
    """data models for features"""
    defaultContext: t.Optional[SpatialContext] = SpatialContext.MAP
    """default spatial context"""
    templates: t.Optional[t.List[gws.ext.config.template]]
    """feature formatting templates"""
    title: t.Optional[str]
    """provider title"""
    tolerance: str = '10px'
    """tolerance, in pixels or map units"""
    withGeometry: bool = True
    """enable geometry search"""
    withKeyword: bool = True
    """enable keyword search"""
    withFilter: bool = True
    """enable filter search"""


class Object(gws.Node, gws.IFinder):
    spatialContext: SpatialContext
    title: str

    supportsFilter = False
    supportsGeometry = False
    supportsKeyword = False

    def configure(self):
        self.templates = []
        self.models = []

        p = self.var('tolerance')
        self.tolerance = gws.lib.uom.parse(p, default=gws.lib.uom.px) if p else _DEFAULT_TOLERANCE

        self.withKeyword = self.supportsKeyword and self.var('withKeyword', default=True)
        self.withGeometry = self.supportsGeometry and self.var('withGeometry', default=True)
        self.withFilter = self.supportsFilter and self.var('withFilter', default=True)

        self.spatialContext = self.var('defaultContext', default=SpatialContext.MAP)
        self.title = self.var('title', default='')

    def configure_models(self):
        p = self.var('models')
        if p:
            self.models = self.create_children(gws.ext.object.model, p)
            return True

    def configure_templates(self):
        self.create_children(gws.ext.object.template, self.var('templates'))
        return True

    def can_run(self, search, user):
        has_param = False

        if search.keyword:
            if not self.withKeyword:
                return False
            has_param = True

        if search.shape:
            if not self.withGeometry:
                return False
            has_param = True

        if search.ogcFilter:
            if not self.withFilter:
                return False
            has_param = True

        return has_param

    def context_shape(self, search: gws.SearchArgs) -> gws.IShape:
        if search.shape:
            return search.shape
        if self.spatialContext == SpatialContext.VIEW and search.bounds:
            return gws.base.shape.from_bounds(search.bounds)
        if search.project:
            return gws.base.shape.from_bounds(search.project.map.bounds)

    def run(self, search, user, layer=None):
        model = gws.base.model.locate(self.models, user, gws.Access.read)
        if not model and layer:
            model = gws.base.model.locate(layer.models, user, gws.Access.read)
        if not model:
            gws.log.debug(f'no model for {user.uid!r} in finder {self.uid!r}')
            return []
        return model.find_features(search, user)


##

def locate(
        finders: t.List[gws.IFinder],
        user: gws.IUser = None,
        uid: str = None
) -> t.Optional[gws.IFinder]:
    for finder in finders:
        if user and user.can_use(finder):
            continue
        if uid and finder.uid != uid:
            continue
        return finder
