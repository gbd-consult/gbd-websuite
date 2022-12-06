import gws
import gws.base.model
import gws.base.template
import gws.base.shape
import gws.lib.uom
import gws.types as t

_DEFAULT_TOLERANCE = 10, gws.Uom.PX


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
        p = self.var('templates')
        self.templateMgr = self.create_child(gws.base.template.manager.Object, gws.Config(
            templates=p)) if p else None

        p = self.var('models')
        self.modelMgr = self.create_child(gws.base.model.manager.Object, gws.Config(
            models=p)) if p else None

        p = self.var('tolerance')
        self.tolerance = gws.lib.uom.parse(p, default=gws.lib.uom.PX) if p else _DEFAULT_TOLERANCE

        self.withKeyword = self.supportsKeyword and self.var('withKeyword', default=True)
        self.withGeometry = self.supportsGeometry and self.var('withGeometry', default=True)
        self.withFilter = self.supportsFilter and self.var('withFilter', default=True)

        self.spatialContext = self.var('defaultContext', default=SpatialContext.MAP)
        self.title = self.var('title', default='')

    def can_run(self, args: gws.SearchArgs):
        has_param = False

        if args.keyword:
            if not self.withKeyword:
                return False
            has_param = True

        if args.shapes:
            if not self.withGeometry:
                return False
            has_param = True

        if args.filter:
            if not self.withFilter:
                return False
            has_param = True

        return has_param

    def context_shape(self, args: gws.SearchArgs) -> gws.IShape:
        if args.shapes:
            return args.shapes[0].union(args.shapes[1:])
        if self.spatialContext == SpatialContext.VIEW and args.bounds:
            return gws.base.shape.from_bounds(args.bounds)
        if args.project:
            return gws.base.shape.from_bounds(args.project.map.bounds)

    def run(self, args, layer):
        return []
