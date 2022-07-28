import gws
import gws.base.model
import gws.base.template
import gws.gis.shape
import gws.lib.units
import gws.types as t

_DEFAULT_PIXEL_TOLERANCE = 10


class SpatialContext(t.Enum):
    map = 'map'  #: search in the map extent
    view = 'view'  #: search in the client view extent


class Config(gws.ConfigWithAccess):
    dataModel: t.Optional[gws.base.model.Config]  #: feature data model
    defaultContext: t.Optional[SpatialContext] = SpatialContext.map  #: default spatial context
    templates: t.Optional[t.List[gws.ext.config.template]]  #: feature formatting templates
    title: t.Optional[str]  #: provider title
    tolerance: str = '10px'  #: tolerance, in pixels or map units
    withGeometry: bool = True  #: enable geometry search
    withKeyword: bool = True  #: enable keyword search
    withFilter: bool = True  #: enable filter search


class Object(gws.Node, gws.ISearchProvider):
    spatial_context: SpatialContext

    def configure(self):
        self.data_model = self.create_child_if_config(gws.base.model.Object, self.var('dataModel'))

        self.templates = None
        p = self.var('templates')
        if p:
            self.templates = gws.base.template.bundle.create(self.root, items=p, parent=self)

        p = self.var('tolerance')
        self.tolerance = (
            gws.lib.units.parse(p, default=gws.lib.units.PX) if p
            else (_DEFAULT_PIXEL_TOLERANCE, gws.lib.units.PX))

        self.with_keyword = self.var('withKeyword', default=True)
        self.with_geometry = self.var('withGeometry', default=True)
        self.with_filter = self.var('withFilter', default=True)

        self.spatial_context = self.var('defaultContext', default=SpatialContext.map)
        self.title = self.var('title', default='')

    def can_run(self, args: gws.SearchArgs):
        has_param = False

        if args.keyword:
            if not self.supports_keyword or not self.with_keyword:
                return False
            has_param = True

        if args.shapes:
            if not self.supports_geometry or not self.with_geometry:
                return False
            has_param = True

        if args.filter:
            if not self.supports_filter or not self.with_filter:
                return False
            has_param = True

        return has_param

    def context_shape(self, args: gws.SearchArgs) -> gws.IShape:
        if args.shapes:
            return gws.gis.shape.union(args.shapes)
        if self.spatial_context == SpatialContext.view and args.bounds:
            return gws.gis.shape.from_bounds(args.bounds)
        if args.project:
            return gws.gis.shape.from_bounds(args.project.map.bounds)
