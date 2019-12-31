import gws
import gws.gis.shape
import gws.common.format
import gws.common.model
import gws.common.template
import gws.types as t


class Config(t.WithTypeAndAccess):
    defaultContext: str = ''  #: default spatial context ('view' or 'map')
    featureFormat: t.Optional[gws.common.template.FeatureFormatConfig]  #: feature formatting options
    dataModel: t.Optional[gws.common.model.Config]  #: feature data model
    geometryRequired: bool = False
    keywordRequired: bool = False


#:export ISearchProvider
class Object(gws.Object, t.ISearchProvider):
    def __init__(self):
        super().__init__()
        self.geometry_required: bool = False
        self.keyword_required: bool = False

    def configure(self):
        super().configure()

        p = self.var('featureFormat')
        if p:
            self.feature_format = self.add_child('gws.common.format', p)

        p = self.var('dataModel')
        if p:
            self.data_model = self.add_child('gws.common.model', p)

        self.keyword_required = self.var('keywordRequired')
        self.geometry_required = self.var('geometryRequired')

    def can_run(self, args: t.SearchArgs):
        if self.keyword_required and not args.keyword:
            return False
        if self.geometry_required and not args.shapes:
            return False
        return args.keyword or args.shapes

    def context_shape(self, args: t.SearchArgs):
        if args.get('shapes'):
            return gws.gis.shape.union(args.get('shapes'))
        ctx = self.var('defaultContext')
        if ctx == 'view' and args.get('bbox'):
            return gws.gis.shape.from_extent(args.bbox, args.crs)
        if ctx == 'map':
            return gws.gis.shape.from_extent(args.project.map.extent, args.crs)

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        return []
