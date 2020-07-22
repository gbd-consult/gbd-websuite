import gws
import gws.gis.shape
import gws.common.format
import gws.common.model
import gws.common.template
import gws.tools.units
import gws.types as t

_DEFAULT_PIXEL_TOLERANCE = 10


class ParameterUsage(t.Enum):
    allowed = 'allowed'
    forbidden = 'forbidden'
    required = 'required'
    ignored = 'ignored'


class Config(t.WithTypeAndAccess):
    dataModel: t.Optional[gws.common.model.Config]  #: feature data model
    defaultContext: str = ''  #: default spatial context ('view' or 'map')
    templates: t.Optional[t.List[t.ext.template.Config]]  #: feature formatting templates
    tolerance: str = '10px'  #: tolerance, in pixels or map units
    withGeometry: ParameterUsage = 'allowed'  #: whether to use geometry with this search
    withKeyword: ParameterUsage = 'allowed'  #: whether to use keywords with this search


#:export ISearchProvider
class Object(gws.Object, t.ISearchProvider):
    def configure(self):
        super().configure()

        # `active` will be False for automatic search providers that failed to initialize
        # @TODO remove inactive prodivers from the tree
        self.active = True

        p = self.var('dataModel')
        self.data_model: t.Optional[t.IModel] = self.create_child('gws.common.model', p) if p else None

        self.templates: t.List[t.ITemplate] = gws.common.template.configure_list(
            self.root, self.var('templates'))

        p = self.var('tolerance')
        self.tolerance: t.Measurement = (
            gws.tools.units.parse(p, units=['px', 'm'], default='px') if p
            else (_DEFAULT_PIXEL_TOLERANCE, 'px'))

        self.with_keyword: ParameterUsage = self.var('withKeyword')
        self.with_geometry: ParameterUsage = self.var('withGeometry')

    def can_run(self, args: t.SearchArgs):
        if not self.active:
            return False

        # usage:           allowed   forbidden   required
        # param present    ok        ERR         ok
        # param missing    ok        ok          ERR

        if args.keyword and self.with_keyword == ParameterUsage.forbidden:
            return False
        if not args.keyword and self.with_keyword == ParameterUsage.required:
            return False
        geom = args.bounds or args.shapes
        if geom and self.with_geometry == ParameterUsage.forbidden:
            return False
        if not geom and self.with_geometry == ParameterUsage.required:
            return False
        return args.keyword or geom

    def context_shape(self, args: t.SearchArgs) -> t.IShape:
        if args.get('shapes'):
            return gws.gis.shape.union(args.shapes)
        ctx = self.var('defaultContext')
        if ctx == 'view' and args.bounds:
            return gws.gis.shape.from_bounds(args.bounds)
        return gws.gis.shape.from_bounds(args.project.map.bounds)

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        return []
