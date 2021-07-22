import io

import gws
import gws.base.metadata
import gws.base.model
import gws.base.search.runner
import gws.base.template
import gws.base.web.error
import gws.lib.bounds
import gws.lib.date
import gws.lib.extent
import gws.lib.gml
import gws.lib.img
import gws.lib.mime
import gws.lib.misc
import gws.lib.proj
import gws.lib.render
import gws.lib.units as units
import gws.lib.xml2
import gws.lib.xml2.helper
import gws.types as t

_DEFAULT_FEATURE_NAME = 'feature'
_DEFAULT_GEOMETRY_NAME = 'geometry'


class Config(gws.WithAccess):
    metaData: t.Optional[gws.base.metadata.Config]  #: service metadata
    root: str = ''  #: root layer uid
    name: str = ''  #: service name
    supportedCrs: t.Optional[t.List[gws.Crs]]  #: supported CRS for this service
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: service XML templates
    updateSequence: t.Optional[str]  #: service update sequence
    withInspireMeta: bool = False  #: use INSPIRE Metadata
    strictParams: bool = False  #: strict parameter parsing
    forceFeatureName: str = ''


class Request(gws.Data):
    req: gws.IWebRequest
    project: gws.IProject
    service: gws.IOwsService
    xml: t.Optional[gws.lib.xml2.Element] = None
    xml_is_soap: bool = False


class Projection(gws.Data):
    crs: str
    proj: gws.lib.proj.Proj
    extent: gws.Extent


class Name(gws.Data):
    p: str  # plain name
    q: str  # qualified name
    ns: str  # namespace
    ns_prefix: str  # namespace prefix
    ns_uri: str  # namespace uri
    ns_schema_location: str  # namespace schema location


class FeatureSchemaAttribute(gws.Data):
    type: str
    name: Name


class LayerCaps(gws.Data):
    layer: gws.ILayer

    has_legend: bool
    has_search: bool
    metadata: gws.IMetaData
    title: str

    layer_name: Name
    feature_name: Name

    extent: gws.Extent
    extent4326: gws.Extent
    max_scale: int
    min_scale: int
    projections: t.List[Projection]

    sub_caps: t.List['LayerCaps']
    feature_schema: t.List[FeatureSchemaAttribute]


class FeatureCaps(gws.Data):
    feature: gws.IFeature
    shape_tag: gws.Tag
    name: Name


class FeatureCollection(gws.Data):
    caps: t.List[FeatureCaps]
    features: t.List[gws.IFeature]
    time_stamp: str
    num_matched: int
    num_returned: int


class Object(gws.Node, gws.IOwsService):
    """Baseclass for OWS services."""

    metadata: gws.IMetaData
    name: str
    service_type: str
    supported_crs: t.List[gws.Crs]
    supported_versions: t.List[str]
    templates: gws.ITemplateBundle
    version: str
    with_inspire_meta: bool
    with_strict_params: bool

    xml_helper: gws.lib.xml2.helper.Object
    project: t.Optional[gws.IProject]

    root_layer_uid: str
    update_sequence: str
    force_feature_name: str

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
        self.metadata = self.configure_metadata()
        self.name = self.var('name') or self.default_name
        self.supported_crs = self.var('supportedCrs', default=[])

        p = self.var('templates')
        self.templates = t.cast(gws.base.template.Bundle, self.create_child(
            gws.base.template.Bundle,
            gws.Config(templates=p)))

        self.with_inspire_meta = self.var('withInspireMeta')
        self.with_strict_params = self.var('withStrictParams')

        self.xml_helper = t.cast(
            gws.lib.xml2.helper.Object,
            self.root.application.helper('xml'))

        self.project = t.cast(gws.IProject, self.get_closest('gws.base.project'))

        self.root_layer_uid = self.var('root')
        self.update_sequence = self.var('updateSequence')
        self.force_feature_name = self.var('forceFeatureName', default='')

    def configure_metadata(self) -> gws.IMetaData:
        m = t.cast(gws.IMetaData, self.create_child(gws.base.metadata.Object, self.var('metaData')))
        m.extend(self.project.metadata if self.project else self.root.application.metadata)

        if not m.data.get('links') and self.service_link:
            m.data.set('links', [self.service_link])

        m.extend(self.default_metadata)
        return m

    # Request handling

    def handle_request(self, req: gws.IWebRequest) -> gws.ContentResponse:
        # services can be configured globally (in which case, self.project == None)
        # and applied to multiple projects with the projectUid param
        # or, configured just for a single project (self.project != None)

        project = None

        p = req.param('projectUid')
        if p:
            project = req.require_project(p)
            if self.project and project != self.project:
                gws.log.debug(f'service={self.uid!r}: wrong project={p!r}')
                raise gws.base.web.error.NotFound('Project not found')
        elif self.project:
            # for in-project services, ensure the user can access the project
            project = req.require_project(self.project.uid)

        rd = Request(req=req, project=project)

        return self.dispatch(rd, req.param('request', ''))

    def dispatch(self, rd: Request, request_param):
        h = getattr(self, 'handle_' + request_param.lower(), None)
        if not h:
            gws.log.debug(f'service={self.uid!r}: request={request_param!r} not found')
            raise gws.base.web.error.BadRequest('Invalid REQUEST parameter')
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

        raise gws.base.web.error.BadRequest('Unsupported service version')

    # Rendering and responses

    def error_response(self, err: Exception):
        status = gws.get(err, 'code') or 500
        description = gws.get(err, 'description') or f'Error {status}'
        return self.xml_error_response(status, description)

    def template_response(self, rd: Request, ows_request: str, ows_format: str = None, context=None):
        out = self.render_template(rd, ows_request, ows_format, context)
        return gws.ContentResponse(content=out.content, mime=out.mime)

    def render_template(self, rd: Request, ows_request: str, ows_format: str = None, context=None, format=None):
        mime = gws.lib.mime.get(ows_format)
        if ows_format and not mime:
            gws.log.debug(f'no mime: ows_request={ows_request!r} ows_format={ows_format!r}')
            raise gws.base.web.error.BadRequest('Invalid FORMAT')

        tpl = self.templates.find(subject='ows.' + ows_request.lower(), mime=mime)
        if not tpl:
            gws.log.debug(f'no template: ows_request={ows_request!r} ows_format={ows_format!r}')
            raise gws.base.web.error.BadRequest('Unsupported FORMAT')

        gws.log.debug(f'ows_request={ows_request!r} ows_format={ows_format!r} template={tpl.uid!r}')

        context = gws.merge({
            'project': rd.project,
            'meta': self.metadata,
            'with_inspire_meta': self.with_inspire_meta,
            'url_for': rd.req.site.url_for,
            'service': self,
            'service_url': self.url_for_project(rd.project),
        }, context)

        return tpl.render(context, gws.TemplateRenderArgs(format=format))

    def enum_template_formats(self):
        fs = {}
        for tpl in self.templates.all():
            for m in tpl.mime_types:
                fs.setdefault(tpl.key, [])
                if m not in fs[tpl.key]:
                    fs[tpl.key].append(m)
        return fs

    def xml_error_response(self, status, description) -> gws.ContentResponse:
        description = gws.lib.xml2.encode(description)
        content = (f'<?xml version="1.0" encoding="UTF-8"?>'
                   + f'<ServiceExceptionReport>'
                   + f'<ServiceException code="{status}">{description}</ServiceException>'
                   + f'</ServiceExceptionReport>')
        # @TODO, check OGC 17-007r1
        # return self.xml_response(content, status)
        return self.xml_response(content, 200)

    def xml_response(self, content, status=200) -> gws.ContentResponse:
        return gws.ContentResponse(
            mime=gws.lib.mime.XML,
            content=gws.lib.xml2.as_string(content),
            status=status,
        )

    # LayerCaps and lists

    def layer_root_caps(self, rd: Request) -> t.Optional[LayerCaps]:
        """Return the root layer caps for a project."""

        def enum(layer_uid):
            layer = t.cast(gws.ILayer, rd.req.acquire('gws.ext.layer', layer_uid))
            if not self.is_layer_enabled(layer):
                return
            sub = []
            if layer.layers:
                sub = gws.compact(enum(la.uid) for la in layer.layers)
            return self._layer_caps(layer, sub)

        if not rd.project:
            return None

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

        layer_names = set()

        for name in names:
            layer_names.add(name)
            # some agents invent their own namespaces, so add non-namespaced versions to the layer names set
            if ':' in name:
                layer_names.add(name.split(':')[1])

        return self.layer_caps_list(rd, layer_names)

    def _layer_caps(self, layer: gws.ILayer, sub_caps=None) -> LayerCaps:

        lc = LayerCaps()

        lc.layer = layer
        lc.title = layer.title
        lc.layer_name = self._parse_name(layer.ows_name)
        lc.feature_name = self._parse_name(self.force_feature_name or layer.ows_feature_name)
        lc.metadata = layer.metadata
        lc.sub_caps = sub_caps or []

        lc.extent = layer.extent
        lc.extent4326 = gws.lib.extent.transform_to_4326(layer.extent, layer.crs)
        lc.has_legend = layer.has_legend or any(s.has_legend for s in lc.sub_caps)
        lc.has_search = layer.has_search or any(s.has_search for s in lc.sub_caps)

        scales = [gws.lib.units.res2scale(r) for r in layer.resolutions]
        lc.min_scale = int(min(scales))
        lc.max_scale = int(max(scales))

        lc.projections = [
            Projection(
                crs=crs,
                proj=gws.lib.proj.as_projection(crs),
                extent=gws.lib.extent.transform(layer.extent, layer.crs, crs)
            )
            for crs in self.supported_crs or [layer.crs]
        ]

        schema = t.cast(gws.base.model.Object, layer.data_model).xml_schema(_DEFAULT_GEOMETRY_NAME)

        lc.feature_schema = [
            FeatureSchemaAttribute(
                type=typ,
                name=self._parse_name(name, lc.feature_name.ns))
            for name, typ in schema.items()
        ]

        return lc

    # FeatureCaps and collections

    def feature_collection(self, features: t.List[gws.IFeature], rd: Request, populate=True, target_crs=None, invert_axis_if_geographic=False, crs_format='uri') -> FeatureCollection:
        coll = FeatureCollection(
            caps=[],
            features=[],
            time_stamp=gws.lib.date.now_iso(with_tz=False),
            num_matched=len(features),
            num_returned=len(features) if populate else 0,
        )

        if not populate:
            return coll

        default_name = self._parse_name(_DEFAULT_FEATURE_NAME)
        prec = 2
        target_proj = None

        if target_crs:
            target_proj = gws.lib.proj.as_proj(target_crs)
            if target_proj and target_proj.is_geographic:
                prec = 6

        for f in features:
            if target_proj:
                f.transform_to(target_proj.epsg)

            gs = None
            if f.shape:
                inv = target_proj and target_proj.is_geographic and invert_axis_if_geographic
                gs = gws.lib.gml.shape_to_tag(f.shape, precision=prec, invert_axis=inv, crs_format=crs_format)

            f.apply_data_model()

            name = self._parse_name(f.layer.ows_feature_name) if f.layer else default_name
            if self.force_feature_name:
                name = self._parse_name(self.force_feature_name)

            coll.caps.append(FeatureCaps(
                feature=f,
                shape_tag=gs,
                name=name
            ))

            coll.features.append(f)

        return coll

    # Utils

    def url_for_project(self, project):
        u = gws.SERVER_ENDPOINT + '/owsService/uid/' + self.uid
        if project:
            u += f'/projectUid/{project.uid}'
        return u

    def render_map_bbox_from_layer_caps_list(self, lcs: t.List[LayerCaps], bounds: gws.Bounds, rd: Request) -> gws.ContentResponse:
        try:
            px_width = int(rd.req.param('width'))
            px_height = int(rd.req.param('height'))
        except:
            raise gws.base.web.error.BadRequest()

        if not bounds or not px_width or not px_height:
            raise gws.base.web.error.BadRequest()

        render_input = gws.MapRenderInput(
            background_color=0 if rd.req.param('transparent', '').lower() == 'false' else None,
            items=[],
            view=gws.lib.render.view_from_bbox(
                crs=bounds.crs,
                bbox=bounds.extent,
                out_size=(px_width, px_height),
                out_size_unit='px',
                rotation=0,
                dpi=0)
        )

        for lc in lcs:
            render_input.items.append(gws.MapRenderInputItem(
                type=gws.MapRenderInputItemType.image_layer,
                layer=lc.layer))

        renderer = gws.lib.render.Renderer()
        for _ in renderer.run(render_input):
            pass

        out = renderer.output
        if not out.items:
            content = gws.lib.misc.Pixels.png8
        else:
            content = gws.lib.img.image_to_bytes(out.items[0].image, format='png')

        return gws.ContentResponse(mime='image/png', content=content)

    def is_layer_enabled(self, layer):
        return layer and layer.enabled_for_ows(self)

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
