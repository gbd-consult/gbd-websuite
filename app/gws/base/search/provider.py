import gws
import gws.base.model
import gws.base.template
import gws.lib.shape
import gws.lib.units
import gws.types as t

_DEFAULT_PIXEL_TOLERANCE = 10


class SpatialContext(t.Enum):
    map = 'map'  #: search in the map extent
    view = 'view'  #: search in the client view extent


class Config(gws.WithAccess):
    dataModel: t.Optional[gws.base.model.Config]  #: feature data model
    defaultContext: t.Optional[SpatialContext] = SpatialContext.map  #: default spatial context
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: feature formatting templates
    title: t.Optional[str]  #: provider title
    tolerance: str = '10px'  #: tolerance, in pixels or map units
    withGeometry: bool = True  #: enable geometry search
    withKeyword: bool = True  #: enable keyword search
    withFilter: bool = True  #: enable filter search


class Object(gws.Object, gws.ISearchProvider):
    supports_keyword: bool = False
    supports_geometry: bool = False
    supports_filter: bool = False

    with_keyword: bool
    with_geometry: bool
    with_filter: bool

    data_model: t.Optional[gws.IDataModel]
    templates: t.Optional[gws.ITemplateBundle]
    tolerance: gws.Measurement

    spatial_context: SpatialContext
    title: str

    def configure(self):
        self.data_model = self.create_child_if_config(gws.base.model.Object, self.var('dataModel'))

        self.templates = None
        p = self.var('templates')
        if p:
            self.templates = gws.base.template.create_bundle(self, p)

        p = self.var('tolerance')
        self.tolerance = (
            gws.lib.units.parse(p, units=['px', 'm'], default='px') if p
            else (_DEFAULT_PIXEL_TOLERANCE, 'px'))

        self.with_keyword = self.supports_keyword and self.var('withKeyword', default=True)
        self.with_geometry = self.supports_geometry and self.var('withGeometry', default=True)
        self.with_filter = self.supports_filter and self.var('withFilter', default=True)

        self.spatial_context = self.var('defaultContext', default=SpatialContext.map)
        self.title = self.var('title', default='')

    def can_run(self, args: gws.SearchArgs):
        if args.keyword and not self.with_keyword:
            return False

        if args.shapes and not self.with_geometry:
            return False

        if args.filter and not self.with_filter:
            return False

        return bool(args.keyword or args.shapes or args.filter)

    def context_shape(self, args: gws.SearchArgs) -> gws.IShape:
        if args.shapes:
            return gws.lib.shape.union(args.shapes)
        if self.spatial_context == SpatialContext.view and args.bounds:
            return gws.lib.shape.from_bounds(args.bounds)
        if args.project:
            return gws.lib.shape.from_bounds(args.project.map.bounds)
