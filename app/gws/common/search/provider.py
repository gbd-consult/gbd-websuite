import gws
import gws.gis.shape
import gws.common.format
import gws.types as t


class Config(t.WithTypeAndAccess):
    defaultContext: str = ''  #: default spatial context ('view' or 'map')
    featureFormat: t.Optional[t.FormatConfig]  #: feature formatting options
    title: str = ''  #: search results title
    uid: str = ''  #: unique ID


class Object(gws.Object, t.SearchProviderInterface):
    def __init__(self):
        super().__init__()
        self.feature_format: t.FormatInterface = None
        self.title = ''

    def configure(self):
        super().configure()

        p = self.var('featureFormat')
        if p:
            self.feature_format = self.create_object('gws.common.format', p)

        self.title = self.var('title', default='')

    def context_shape(self, args: t.SearchArgs):
        if args.get('shapes'):
            return gws.gis.shape.union(args.get('shapes'))
        ctx = self.var('defaultContext')
        if ctx == 'view' and args.get('bbox'):
            return gws.gis.shape.from_bbox(args.bbox, args.crs)
        if ctx == 'map':
            return gws.gis.shape.from_bbox(args.project.map.extent, args.crs)
