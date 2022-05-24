import gws
import gws.gis.shape
import gws.common.model
import gws.common.template
import gws.tools.units
import gws.types as t

_DEFAULT_PIXEL_TOLERANCE = 10


#:export
class SearchSpatialContext(t.Enum):
    map = 'map'  #: search in the map extent
    view = 'view'  #: search in the client view extent


class Config(t.WithTypeAndAccess):
    model: t.Optional[gws.common.model.Config]  #: feature data model
    modelUid: t.Optional[str]  #: feature data model
    defaultContext: t.Optional[SearchSpatialContext] = 'map'  #: default spatial context
    templates: t.Optional[t.List[t.ext.template.Config]]  #: feature formatting templates
    title: t.Optional[str]  #: provider title
    category: t.Optional[str]  #: provider category
    tolerance: str = '10px'  #: tolerance, in pixels or map units
    withGeometry: bool = True  #: enable geometry search
    withKeyword: bool = True  #: enable keyword search


CAPS_KEYWORD = 1 << 0
CAPS_GEOMETRY = 1 << 1
CAPS_FILTER = 1 << 2


#:export ISearchProvider
class Object(gws.Object, t.ISearchProvider):
    def configure(self):
        super().configure()

        self.capabilties = 0  # must be a sum of CAPS flags

        # `active` will be False for automatic search providers that failed to initialize
        # @TODO remove inactive prodivers from the tree
        self.active = True

        p = self.var('model')
        self.model: t.Optional[t.IModel] = self.create_child('gws.common.model', p) if p else None

        self.templates: t.List[t.ITemplate] = gws.common.template.bundle(self, self.var('templates'))

        p = self.var('tolerance')
        self.tolerance: t.Measurement = (
            gws.tools.units.parse(p, units=['px', 'm'], default='px') if p
            else (_DEFAULT_PIXEL_TOLERANCE, 'px'))

        self.with_keyword: bool = self.var('withKeyword', default=True)
        self.with_geometry: bool = self.var('withGeometry', default=True)
        self.spatial_context: SearchSpatialContext = self.var('defaultContext', default=SearchSpatialContext.map)
        self.title: str = self.var('title', default='')
        self.category: str = self.var('category', default=self.title)

    def post_configure(self):
        p = self.var('modelUid')
        if p:
            self.model = gws.common.model.registry().get_model(p)

    @property
    def supports_keyword(self):
        return CAPS_KEYWORD & self.capabilties

    @property
    def supports_geometry(self):
        return CAPS_GEOMETRY & self.capabilties

    @property
    def categories(self):
        return [self.title]

    def can_run(self, args: t.SearchArgs):
        if not self.active:
            gws.log.debug('can_run: inactive')
            return False

        if args.keyword:
            if not (CAPS_KEYWORD & self.capabilties):
                gws.log.debug('can_run: no keyword caps')
                return False
            if not self.with_keyword:
                gws.log.debug('can_run: with_keyword=false')
                return False

        if args.shapes:
            if not (CAPS_GEOMETRY & self.capabilties):
                gws.log.debug('can_run: no geometry caps')
                return False
            if not self.with_geometry:
                gws.log.debug('can_run: with_geometry=false')
                return False

        if args.filter:
            if not (CAPS_FILTER & self.capabilties):
                gws.log.debug('can_run: no filter caps')
                return False

        if args.keyword or args.shapes or args.filter:
            return True

        gws.log.debug('can_run: not enough args')
        return False

    def context_shape(self, args: t.SearchArgs) -> t.IShape:
        if args.get('shapes'):
            return gws.gis.shape.union(args.shapes)
        if self.spatial_context == SearchSpatialContext.view and args.bounds:
            return gws.gis.shape.from_bounds(args.bounds)
        return gws.gis.shape.from_bounds(args.project.map.bounds)

    def run(self, req: t.IRequest, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        return []
