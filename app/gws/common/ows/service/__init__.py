import io

import gws
import gws.common.metadata
import gws.common.metadata.inspire
import gws.common.model
import gws.common.search.runner
import gws.common.template
import gws.gis.bounds
import gws.gis.extent
import gws.gis.render
import gws.gis.renderview
import gws.gis.gml
import gws.gis.proj
import gws.tools.units as units
import gws.tools.xml2
import gws.tools.date
import gws.tools.mime
import gws.tools.misc
import gws.web.error
import gws.ext.helper.xml

import gws.types as t

_XML_SCHEMA_TYPES = {
    t.AttributeType.bool: 'xsd:boolean',
    t.AttributeType.bytes: None,
    t.AttributeType.date: 'xsd:date',
    t.AttributeType.datetime: 'datetime',
    t.AttributeType.float: 'xsd:decimal',
    t.AttributeType.geometry: None,
    t.AttributeType.int: 'xsd:integer',
    t.AttributeType.list: None,
    t.AttributeType.str: 'xsd:string',
    t.AttributeType.text: 'xsd:string',
    t.AttributeType.time: 'xsd:time',
    t.GeometryType.curve: 'gml:CurvePropertyType',
    t.GeometryType.geomcollection: 'gml:MultiGeometryPropertyType',
    t.GeometryType.geometry: 'gml:MultiGeometryPropertyType',
    t.GeometryType.linestring: 'gml:CurvePropertyType',
    t.GeometryType.multicurve: 'gml:MultiCurvePropertyType',
    t.GeometryType.multilinestring: 'gml:MultiCurvePropertyType',
    t.GeometryType.multipoint: 'gml:MultiPointPropertyType',
    t.GeometryType.multipolygon: 'gml:MultiPolygonPropertyType',
    t.GeometryType.multisurface: 'gml:MultiGeometryPropertyType',
    t.GeometryType.point: 'gml:PointPropertyType',
    t.GeometryType.polygon: 'gml:SurfacePropertyType',
    t.GeometryType.polyhedralsurface: 'gml:SurfacePropertyType',
    t.GeometryType.surface: 'gml:SurfacePropertyType',
}

_DEFAULT_FEAUTURE_NAME = 'feature'
_DEFAULT_GEOMETRY_NAME = 'geometry'


class Config(t.WithTypeAndAccess):
    meta: t.Optional[gws.common.metadata.Config]  #: service metadata
    root: str = ''  #: root layer uid
    name: str = ''  #: service name
    supportedCrs: t.Optional[t.List[t.Crs]]  #: supported CRS for this service
    templates: t.Optional[t.List[t.ext.template.Config]]  #: service XML templates
    updateSequence: t.Optional[str]  #: service update sequence
    withInspireMeta: bool = False  #: use INSPIRE Metadata


class Request(t.Data):
    req: t.IRequest
    project: t.IProject
    service: t.IOwsService
    xml: gws.tools.xml2.Element = None
    xml_is_soap: bool = False


class Projection(t.Data):
    crs: str
    proj: gws.gis.proj.Proj
    extent: t.Extent


class Name(t.Data):
    p: str  # plain name
    q: str  # qualified name
    ns: str  # namespace
    ns_prefix: str  # namespace prefix
    ns_uri: str  # namespace uri
    ns_schema_location: str  # namespace schema location


class FeatureSchemaAttribute(t.Data):
    type: str
    name: Name


class LayerCaps(t.Data):
    layer: t.ILayer

    has_legend: bool
    has_search: bool
    meta: t.MetaData
    title: str

    layer_name: Name
    feature_name: Name

    extent: t.Extent
    extent4326: t.Extent
    max_scale: int
    min_scale: int
    projections: t.List[Projection]

    sub_caps: t.List['LayerCaps']
    feature_schema: t.List[FeatureSchemaAttribute]


class FeatureCaps(t.Data):
    feature: t.IFeature
    shape_tag: t.Tag
    name: Name


class FeatureCollection(t.Data):
    caps: t.List[FeatureCaps]
    time_stamp: str
    num_matched: int
    num_returned: int


#:export IOwsService
class Object(gws.Object, t.IOwsService):
    """OWS service interface."""

    def configure(self):
        super().configure()

        self.type = ''
        self.version = ''
        self.meta: t.MetaData = t.MetaData()

    def handle(self, req: t.IRequest) -> t.HttpResponse:
        pass

    def error_response(self, err: Exception) -> t.HttpResponse:
        pass


class Base(Object):
    """Baseclass for OWS services."""

    @property
    def service_link(self):
        # NB: for project-based services, e.g. WMS,
        # a service link only makes sense with a bound project
        return None

    @property
    def default_templates(self):
        return []

    @property
    def default_metadata(self):
        return {}

    @property
    def default_name(self):
        return ''

    # Configuration

    def configure(self):
        super().configure()

        self.name = self.var('name') or self.default_name
        self.supported_versions = []

        self.xml_helper: gws.ext.helper.xml.Object = t.cast(
            gws.ext.helper.xml.Object,
            self.root.application.require_helper('xml'))

        self.project: t.Optional[t.IProject] = t.cast(t.IProject, self.get_closest('gws.common.project'))

        self.meta: t.MetaData = self.configure_metadata()

        self.root_layer_uid = self.var('root')
        self.supported_crs: t.List[t.Crs] = self.var('supportedCrs', default=[])
        self.update_sequence = self.var('updateSequence')
        self.with_inspire_meta = self.var('withInspireMeta')

        self.templates: t.List[t.ITemplate] = gws.common.template.bundle(self, self.var('templates'), self.default_templates)

    def configure_metadata(self):
        meta = gws.common.metadata.from_config(self.var('meta'))
        if self.project:
            meta = gws.common.metadata.extend(meta, self.project.meta)
        else:
            meta = gws.common.metadata.extend(meta, self.root.application.meta)

        meta = gws.extend(
            meta,
            catalogUid=self.uid,
            links=[],
        )

        if self.service_link:
            meta.links.append(self.service_link)

        meta = gws.extend(meta, self.default_metadata)
        return meta

    # Request handling

    def handle(self, req) -> t.HttpResponse:
        # services can be configured globally (in which case, self.project == None)
        # and applied to multiple projects with the projectUid param
        # or, configured just for a single project (self.project != None)

        project = None

        p = req.param('projectUid')
        if p:
            project = req.require_project(p)
            if self.project and project != self.project:
                gws.log.debug(f'service={self.uid!r}: wrong project={p!r}')
                raise gws.web.error.NotFound('Project not found')
        elif self.project:
            # for in-project services, ensure the user can access the project
            project = req.require_project(self.project.uid)

        rd = Request(req=req, project=project)

        return self.dispatch(rd, req.param('request', ''))

    def dispatch(self, rd: Request, request_param):
        h = getattr(self, 'handle_' + request_param.lower(), None)
        if not h:
            gws.log.debug(f'service={self.uid!r}: request={request_param!r} not found')
            raise gws.web.error.BadRequest('Invalid REQUEST parameter')
        return h(rd)

    def request_version(self, rd: Request) -> str:
        version = rd.req.param('version') or rd.req.param('acceptversions')
        if version:
            for v in gws.as_list(version):
                for ver in self.supported_versions:
                    if ver.startswith(v):
                        return ver
        elif self.supported_versions:
            # the first supported version is the default
            return self.supported_versions[0]

        raise gws.web.error.BadRequest('Unsupported service version')

    # Rendering and responses

    def error_response(self, err: Exception):
        status = gws.get(err, 'code') or 500
        description = gws.get(err, 'description') or f'Error {status}'
        return self.xml_error_response(status, description)

    def template_response(self, rd: Request, ows_request: str, ows_format: str = None, context=None):
        out = self.render_template(rd, ows_request, ows_format, context)
        return t.HttpResponse(content=out.content, mime=out.mime)

    def render_template(self, rd: Request, ows_request: str, ows_format: str = None, context=None, format=None):
        mime = gws.tools.mime.get(ows_format)
        if ows_format and not mime:
            raise gws.web.error.BadRequest('Invalid FORMAT')
        tpl = gws.common.template.find(self.templates, subject='ows.' + ows_request.lower(), mime=mime)
        if not tpl:
            raise gws.web.error.BadRequest('Unsupported FORMAT')
        gws.log.debug(f'ows_request={ows_request!r} ows_format={ows_format!r} template={tpl.uid!r}')

        context = gws.merge({
            'project': rd.project,
            'meta': self.meta,
            'with_inspire_meta': self.with_inspire_meta,
            'url_for': rd.req.url_for,
            'service': self,
            'service_url': self.url_for_project(rd.project),
        }, context)

        return tpl.render(context, format=format)

    def enum_template_formats(self):
        fs = {}
        for tpl in self.templates:
            for m in tpl.mime_types:
                fs.setdefault(tpl.key, [])
                if m not in fs[tpl.key]:
                    fs[tpl.key].append(m)
        return fs

    def xml_error_response(self, status, description) -> t.HttpResponse:
        description = gws.tools.xml2.encode(description)
        content = (f'<?xml version="1.0" encoding="UTF-8"?>'
                   + f'<ServiceExceptionReport>'
                   + f'<ServiceException code="{status}">{description}</ServiceException>'
                   + f'</ServiceExceptionReport>')
        return self.xml_response(content, status)

    def xml_response(self, content, status=200) -> t.HttpResponse:
        return t.HttpResponse(
            mime=gws.tools.mime.get('xml'),
            content=gws.tools.xml2.as_string(content),
            status=status,
        )

    # LayerCaps and lists

    def layer_root_caps(self, rd: Request) -> t.Optional[LayerCaps]:
        """Return the root layer caps for a project."""

        def enum(layer_uid):
            layer = t.cast(t.ILayer, rd.req.acquire('gws.ext.layer', layer_uid))
            if not self.is_layer_enabled(layer):
                return
            sub = []
            if layer.layers:
                sub = gws.compact(enum(la.uid) for la in layer.layers)
            return self._layer_caps(layer, sub)

        if not rd.project:
            return

        if self.root_layer_uid:
            root = enum(self.root_layer_uid)
        else:
            # no root given, take the first (enabled) root layer
            roots = gws.compact(enum(la.uid) for la in rd.project.map.layers)
            root = roots[0] if roots else None

        if root:
            return root

    def layer_caps_list(self, rd: Request, layer_names=None) -> t.List[LayerCaps]:
        """Return a list of terminal layer caps (for WFS)."""

        lcs = []

        def walk(lc: LayerCaps, names=None):
            # if a group matches, collect all children unconditionally
            # if a terminal matches - add it
            # if a group doesn't match, collect its children conditionally

            matches = (
                    not names
                    or lc.layer_name.p in names
                    or lc.layer_name.q in names
                    or lc.feature_name.p in names
                    or lc.feature_name.q in names
            )

            if matches:
                if lc.sub_caps:
                    for s in lc.sub_caps:
                        walk(s)
                else:
                    lcs.append(lc)
            elif lc.sub_caps:
                for s in lc.sub_caps:
                    walk(s, names)

        root = self.layer_root_caps(rd)
        if root:
            walk(root, layer_names)
        return lcs

    def layer_caps_list_from_request(self, rd: Request, param_names, fallback_to_all=True) -> t.List[LayerCaps]:
        """Return a list of terminal layer caps matching request parameters."""

        names = None

        for p in param_names:
            if rd.req.has_param(p):
                names = gws.as_list(rd.req.param(p))
                break

        if names is None and fallback_to_all:
            return self.layer_caps_list(rd)

        if not names:
            return []

        return self.layer_caps_list(rd, set(names))

    def _layer_caps(self, layer: t.ILayer, sub_caps=None) -> LayerCaps:

        lc = LayerCaps()

        lc.layer = layer
        lc.title = layer.title
        lc.layer_name = self._parse_name(layer.ows_name)
        lc.feature_name = self._parse_name(layer.ows_feature_name)
        lc.meta = layer.meta
        lc.sub_caps = sub_caps or []

        lc.extent = layer.extent
        lc.extent4326 = gws.gis.extent.transform_to_4326(layer.extent, layer.crs)
        lc.has_legend = layer.has_legend or any(s.has_legend for s in lc.sub_caps)
        lc.has_search = layer.has_search or any(s.has_search for s in lc.sub_caps)

        scales = [gws.tools.units.res2scale(r) for r in layer.resolutions]
        lc.min_scale = int(min(scales))
        lc.max_scale = int(max(scales))

        lc.projections = [
            Projection(
                crs=crs,
                proj=gws.gis.proj.as_projection(crs),
                extent=gws.gis.extent.transform(layer.extent, layer.crs, crs)
            )
            for crs in self.supported_crs or [layer.crs]
        ]

        lc.feature_schema = []
        dm = layer.data_model

        if dm:
            for rule in dm.rules:
                x = _XML_SCHEMA_TYPES.get(rule.type)
                if x:
                    lc.feature_schema.append(FeatureSchemaAttribute(
                        type=x,
                        name=self._parse_name(rule.name, lc.feature_name.ns)))

            if dm.geometry_type:
                lc.feature_schema.append(FeatureSchemaAttribute(
                    type=_XML_SCHEMA_TYPES.get(dm.geometry_type),
                    name=self._parse_name(_DEFAULT_GEOMETRY_NAME, lc.feature_name.ns)))

        return lc

    # FeatureCaps

    def feature_collection(self, features: t.List[t.IFeature], rd: Request) -> FeatureCollection:
        coll = FeatureCollection(
            caps=[],
            time_stamp=gws.tools.date.now_iso(with_tz=False),
            num_matched=len(features),
            num_returned=len(features),
        )

        default_name = self._parse_name(_DEFAULT_FEAUTURE_NAME)

        for f in features:
            gs = None
            if f.shape:
                gs = gws.gis.gml.shape_to_tag(f.shape, precision=rd.project.map.coordinate_precision)

            f.apply_data_model()

            coll.caps.append(FeatureCaps(
                feature=f,
                shape_tag=gs,
                name=self._parse_name(f.layer.ows_feature_name) if f.layer else default_name,
            ))

        return coll

    # Utils

    def url_for_project(self, project):
        u = gws.SERVER_ENDPOINT + '/cmd/owsHttpService/uid/' + self.uid
        if project:
            u += f'/projectUid/{project.uid}'
        return u

    def render_map_bbox_from_layer_caps_list(self, lcs: t.List[LayerCaps], rd: Request) -> t.HttpResponse:
        crs = rd.req.param('crs') or rd.req.param('srs') or rd.project.map.crs
        try:
            bounds = gws.gis.bounds.from_request_bbox(rd.req.param('bbox'), crs)
            px_width = int(rd.req.param('width'))
            px_height = int(rd.req.param('height'))
        except:
            raise gws.web.error.BadRequest()

        if not bounds or not px_width or not px_height:
            raise gws.web.error.BadRequest()

        render_input = t.MapRenderInput(
            background_color=None,
            items=[],
            view=gws.gis.renderview.from_bbox(
                crs=bounds.crs,
                bbox=bounds.extent,
                out_size=(px_width, px_height),
                out_size_unit='px',
                rotation=0,
                dpi=0)
        )

        for lc in lcs:
            render_input.items.append(t.MapRenderInputItem(
                type=t.MapRenderInputItemType.image_layer,
                layer=lc.layer))

        renderer = gws.gis.render.Renderer()
        for _ in renderer.run(render_input):
            pass

        out = renderer.output
        if not out.items:
            img = gws.tools.misc.Pixels.png8
        else:
            buf = io.BytesIO()
            out.items[0].image.save(buf, format='png')
            img = buf.getvalue()

        return t.HttpResponse(mime='image/png', content=img)

    def is_layer_enabled(self, layer):
        return layer and layer.ows_enabled(self)

    def _parse_name(self, name, nsid=None) -> Name:
        if ':' in name:
            nsid, name = name.split(':')

        ns = self.xml_helper.namespace(nsid) if nsid else None

        if ns:
            return Name(
                p=name,
                q=nsid + ':' + name,
                ns=nsid,
                ns_prefix=nsid + ':',
                ns_uri=ns[0],
                ns_schema_location=ns[1],
            )
        else:
            return Name(
                p=name,
                q=name,
                ns='',
                ns_prefix='',
                ns_uri='',
                ns_schema_location='',
            )
