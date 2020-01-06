import gws
import gws.gis.shape
import gws.common.format
import gws.common.model
import gws.common.template
import gws.types as t


class ParameterUsage(t.Enum):
    yes = 'yes'
    no = 'no'
    required = 'required'
    ignored = 'ignored'


class Config(t.WithTypeAndAccess):
    defaultContext: str = ''  #: default spatial context ('view' or 'map')
    featureFormat: t.Optional[gws.common.template.FeatureFormatConfig]  #: feature formatting options
    dataModel: t.Optional[gws.common.model.Config]  #: feature data model
    pixelTolerance: int = 5
    withGeometry: ParameterUsage = 'yes'
    withKeyword: ParameterUsage = 'yes'


#:export ISearchProvider
class Object(gws.Object, t.ISearchProvider):
    def __init__(self):
        super().__init__()

        self.with_geometry = ''
        self.with_keyword = ''

        self.pixel_tolerance: int = 0

        self.feature_format: t.IFormat = None
        self.data_model: t.IModel = None

    def configure(self):
        super().configure()

        p = self.var('featureFormat')
        if p:
            self.feature_format = self.add_child('gws.common.format', p)

        p = self.var('dataModel')
        if p:
            self.data_model = self.add_child('gws.common.model', p)

        self.with_keyword = self.var('withKeyword')
        self.with_geometry = self.var('withGeometry')

        self.pixel_tolerance = self.var('pixelTolerance')

    # Parameter usage:
    #          yes  no   required
    # present  ok   ERR  ok
    # missing  ok   ok   ERR

    def can_run(self, args: t.SearchArgs):
        if args.keyword and self.with_keyword == 'no':
            return False
        if not args.keyword and self.with_keyword == 'required':
            return False
        geom = args.bounds or args.shapes
        if geom and self.with_geometry == 'no':
            return False
        if not geom and self.with_geometry == 'required':
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
