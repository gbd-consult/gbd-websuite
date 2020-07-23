import gws
import gws.gis.shape
import gws.common.model
import gws.common.template
import gws.tools.units
import gws.types as t

_DEFAULT_PIXEL_TOLERANCE = 10


class Config(t.WithTypeAndAccess):
    dataModel: t.Optional[gws.common.model.Config]  #: feature data model
    defaultContext: str = ''  #: default spatial context ('view' or 'map')
    templates: t.Optional[t.List[t.ext.template.Config]]  #: feature formatting templates
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

        p = self.var('dataModel')
        self.data_model: t.Optional[t.IModel] = self.create_child('gws.common.model', p) if p else None

        self.templates: t.List[t.ITemplate] = gws.common.template.configure_list(self.root, self.var('templates'))

        p = self.var('tolerance')
        self.tolerance: t.Measurement = (
            gws.tools.units.parse(p, units=['px', 'm'], default='px') if p
            else (_DEFAULT_PIXEL_TOLERANCE, 'px'))

        self.with_keyword: bool = self.var('withKeyword', default=True)
        self.with_geometry: bool = self.var('withGeometry', default=True)

    def can_run(self, args: t.SearchArgs):
        if not self.active:
            return False

        if args.keyword:
            if not (CAPS_KEYWORD & self.capabilties) or not self.with_keyword:
                return False

        geom = args.bounds or args.shapes
        if geom:
            if not (CAPS_GEOMETRY & self.capabilties) or not self.with_geometry:
                return False

        if args.filter:
            if not (CAPS_GEOMETRY & self.capabilties):
                return False

        return bool(args.keyword or geom or args.filter)

    def context_shape(self, args: t.SearchArgs) -> t.IShape:
        if args.get('shapes'):
            return gws.gis.shape.union(args.shapes)
        ctx = self.var('defaultContext')
        if ctx == 'view' and args.bounds:
            return gws.gis.shape.from_bounds(args.bounds)
        return gws.gis.shape.from_bounds(args.project.map.bounds)

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        return []
