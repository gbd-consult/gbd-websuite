import gws
import gws.gis.shape
import gws.common.format
import gws.types as t


class Config(t.WithTypeAndAccess):
    defaultContext: str = ''  #: default spatial context ('view' or 'map')
    featureFormat: t.Optional[t.FeatureFormatConfig]  #: feature formatting options
    dataModel: t.Optional[t.DataModelConfig]  #: feature data model


class Object(gws.Object, t.SearchProviderObject):
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
            self.data_model = self.add_child('gws.common.datamodel', p)

        self.keyword_required = self.var('keywordRequired')
        self.geometry_required = self.var('geometryRequired')

    def context_shape(self, args: t.SearchArguments):
        if args.get('shapes'):
            return gws.gis.shape.union(args.get('shapes'))
        ctx = self.var('defaultContext')
        if ctx == 'view' and args.get('bbox'):
            return gws.gis.shape.from_bbox(args.bbox, args.crs)
        if ctx == 'map':
            return gws.gis.shape.from_bbox(args.project.map.extent, args.crs)
