import gws
import gws.base.model
import gws.base.template
import gws.base.shape
import gws.lib.uom
import gws.types as t

_DEFAULT_PIXEL_TOLERANCE = 10


class SpatialContext(t.Enum):
    map = 'map' 
    """search in the map extent"""
    view = 'view' 
    """search in the client view extent"""


class Config(gws.ConfigWithAccess):
    dataModel: t.Optional[gws.base.model.Config] 
    """feature data model"""
    defaultContext: t.Optional[SpatialContext] = SpatialContext.map 
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
        # self.data_model = self.root.create_optional(gws.base.model.Object, self.var('dataModel'))

        self.templateMgr = None
        p = self.var('templates')
        if p:
            self.templateMgr = self.create_child(gws.base.template.manager.Object, gws.Config(items=p))

        self.tolerance = _DEFAULT_PIXEL_TOLERANCE, gws.lib.uom.PX
        p = self.var('tolerance')
        if p:
            self.tolerance = gws.lib.uom.parse(p, default=gws.lib.uom.PX)

        self.withKeyword = self.supportsKeyword and self.var('withKeyword', default=True)
        self.withGeometry = self.supportsGeometry and self.var('withGeometry', default=True)
        self.withFilter = self.supportsFilter and self.var('withFilter', default=True)

        self.spatialContext = self.var('defaultContext', default=SpatialContext.map)
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
            return gws.base.shape.union(args.shapes)
        if self.spatialContext == SpatialContext.view and args.bounds:
            return gws.base.shape.from_bounds(args.bounds)
        if args.project:
            return gws.base.shape.from_bounds(args.project.map.bounds())
