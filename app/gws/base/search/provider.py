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


class Object(gws.Object, gws.ISearchProvider):
    supports_keyword: bool = False
    supports_geometry: bool = False
    supports_filter: bool = False

    with_keyword: bool = False
    with_geometry: bool = False
    with_filter: bool = False

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
            self.templates = t.cast(
                gws.base.template.Bundle,
                self.create_child(gws.base.template.Bundle, gws.Config(templates=p)))

        p = self.var('tolerance')
        self.tolerance = (
            gws.lib.units.parse(p, units=['px', 'm'], default='px') if p
            else (_DEFAULT_PIXEL_TOLERANCE, 'px'))

        self.with_keyword = self.var('withKeyword', default=True)
        self.with_geometry = self.var('withGeometry', default=True)

        self.spatial_context = self.var('defaultContext', default=SpatialContext.map)
        self.title = self.var('title', default='')

    def can_run(self, p: gws.SearchArgs):
        if p.keyword and (not self.supports_keyword or not self.with_keyword):
            return False

        if p.shapes and (not self.supports_geometry or not self.with_geometry):
            return False

        if p.filter and not self.supports_filter:
            return False

        return bool(p.keyword or p.shapes or p.filter)

    def context_shape(self, p: gws.SearchArgs) -> gws.IShape:
        if p.shapes:
            return gws.lib.shape.union(p.shapes)
        if self.spatial_context == SpatialContext.view and p.bounds:
            return gws.lib.shape.from_bounds(p.bounds)
        if p.project:
            return gws.lib.shape.from_bounds(p.project.map.bounds)
