import re

import gws
import gws.base.template
import gws.base.web.error
import gws.lib.date
import gws.lib.extent
import gws.lib.gml
import gws.lib.image
import gws.lib.metadata
import gws.lib.mime
import gws.lib.proj
import gws.lib.render
import gws.lib.units as units
import gws.lib.xml2
import gws.types as t


class Error(gws.Error):
    pass


class LayerFilter(gws.Data):
    """Layer filter"""

    level: int = 0  #: match only layers at this level
    uids: t.Optional[t.List[str]]  #: match these layer uids
    pattern: gws.Regex = ''  #: match layers whose uid matches a pattern


class LayerConfig(gws.Config):
    """Layer-specific OWS configuration"""

    applyTo: t.Optional[LayerFilter]  #: project layers this configuration applies to
    enabled: bool = True  #: layer is enabled for this service
    layerName: t.Optional[str]  #: layer name for this service
    layerTitle: t.Optional[str]  #: layer title for this service
    featureName: t.Optional[str]  #: feature name for this service


class ServiceConfig(gws.WithAccess):
    layerConfig: t.Optional[t.List[LayerConfig]]  #: custom configurations for specific layers
    metadata: t.Optional[gws.lib.metadata.Config]  #: service metadata
    rootLayer: str = ''  #: root layer uid
    strictParams: bool = False  #: strict parameter parsing
    supportedCrs: t.Optional[t.List[gws.Crs]]  #: supported CRS for this service
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: service XML templates
    updateSequence: t.Optional[str]  #: service update sequence
    withInspireMeta: bool = False  #: use INSPIRE Metadata


##


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


class XmlName(gws.Data):
    p: str  # plain name
    q: str  # qualified name
    ns: gws.lib.xml2.Namespace  # namespace


class FeatureSchemaAttribute(gws.Data):
    type: str
    xname: XmlName


class LayerOptions(gws.Data):
    level: int
    uids: t.Set[str]
    pattern: str
    enabled: bool
    layer_name: str
    layer_title: str
    feature_name: str


class LayerCaps(gws.Data):
    layer: gws.ILayer

    has_legend: bool
    has_search: bool
    meta: gws.lib.metadata.Values
    title: str

    layer_xname: XmlName
    feature_xname: XmlName

    extent: gws.Extent
    extent4326: gws.Extent
    max_scale: int
    min_scale: int
    projections: t.List[Projection]

    children: t.List['LayerCaps']
    ancestors: t.List['LayerCaps']

    adhoc_feature_schema: t.Optional[t.List[FeatureSchemaAttribute]]


class LayerCapsTree(gws.Data):
    roots: t.List[LayerCaps]
    leaves: t.List['LayerCaps']


class FeatureCaps(gws.Data):
    feature: gws.IFeature
    shape_tag: gws.Tag
    xname: XmlName


class FeatureCollection(gws.Data):
    caps: t.List[FeatureCaps]
    features: t.List[gws.IFeature]
    time_stamp: str
    num_matched: int
    num_returned: int


##


class Service(gws.Node, gws.IOwsService):
    """Baseclass for OWS services."""

    xml_helper: gws.lib.xml2.helper.Object
    project: t.Optional[gws.IProject]

    root_layer_uid: str
    update_sequence: str
    layer_options: t.List[LayerOptions]

    is_raster_ows: bool = False
    is_vector_ows: bool = False

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

    # Configuration

    def configure(self):
        self.project = self.get_closest('gws.base.project')
        self.metadata = self.configure_metadata()

        self.supported_crs = self.var('supportedCrs', default=[])
        self.update_sequence = self.var('updateSequence')
        self.with_inspire_meta = self.var('withInspireMeta')
        self.with_strict_params = self.var('withStrictParams')
        self.xml_helper = self.root.application.require_helper('xml')
        self.root_layer_uid = self.var('rootLayer')

        self.templates = gws.base.template.bundle.create(
            self.root,
            gws.Config(
                templates=self.var('templates'),
                defaults=self.default_templates
            ),
            parent=self)

        self.layer_options = []
        for cfg in self.var('layerConfig', default=[]):
            lo = LayerOptions()
            apply = cfg.applyTo or gws.Data()
            lo.uids = set(apply.uids or [])
            lo.level = apply.level
            lo.pattern = apply.pattern
            lo.enabled = cfg.enabled
            lo.layer_name = cfg.layerName
            lo.feature_name = cfg.featureName
            self.layer_options.append(lo)

    def configure_metadata(self) -> gws.lib.metadata.Metadata:
        m = gws.lib.metadata.from_config(self.var('metadata'))
        m.extend(
            self.project.metadata if self.project else self.root.application.metadata,
            self.default_metadata,
            {'name': self.protocol})

        if not m.get('extraLinks') and self.service_link:
            m.set('extraLinks', [self.service_link])

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
            for v in gws.to_list(version):
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

    def template_response(self, rd: Request, verb: gws.OwsVerb, format: str = None, context=None):
        out = self.render_template(rd, verb, format, context)
        return gws.ContentResponse(content=out.content, mime=out.mime)

    def render_template(self, rd: Request, verb: gws.OwsVerb, format: str = None, context=None):
        mime = None

        if format:
            mime = gws.lib.mime.get(format)
            if not mime:
                gws.log.debug(f'no mimetype: verb={verb!r} format={format!r}')
                raise gws.base.web.error.BadRequest('Invalid FORMAT')

        tpl = self.templates.find(category='ows', name=str(verb), mime=mime)
        if not tpl:
            gws.log.debug(f'no template: verb={verb!r} format={format!r}')
            raise gws.base.web.error.BadRequest('Unsupported FORMAT')

        gws.log.debug(f'template found: verb={verb!r} format={format!r} tpl={tpl.uid!r}')

        context = gws.merge({
            'project': rd.project,
            'service': self,
            'service_meta': self.metadata.values,
            'service_url': rd.req.url_for(self.service_url_path(rd.project)),
            'url_for': rd.req.url_for,
            'with_inspire_meta': self.with_inspire_meta,
        }, context)

        return tpl.render({'ARGS': context})

    def enum_template_formats(self):
        fs = {}

        for tpl in self.templates.items:
            for mime in tpl.mime_types:
                fs.setdefault(tpl.name, set())
                fs[tpl.name].add(mime)

        return {k: sorted(v) for k, v in fs.items()}

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
            content=gws.lib.xml2.to_string(content),
            status=status,
        )

    # LayerCaps and lists

    # @TODO should be cached for public layers

    def layer_caps_tree(self, rd: Request) -> LayerCapsTree:
        """Return the root layer caps for a project."""

        def enum(layer_uid, ancestors):
            layer: gws.ILayer = rd.req.acquire('gws.ext.layer', layer_uid)
            if not layer:
                return

            opts = self._layer_options(layer, level=len(ancestors) + 1)
            if not opts:
                return

            lc = LayerCaps()
            lc.ancestors = ancestors + [lc]

            if not layer.layers:
                self._populate_layer_caps(lc, layer, opts, [])
                tree.leaves.append(lc)
                return lc

            ch = gws.compact(enum(la.uid, lc.ancestors) for la in layer.layers)
            if ch:
                self._populate_layer_caps(lc, layer, opts, ch)
                return lc

        tree = LayerCapsTree(leaves=[], roots=[])

        if not rd.project:
            return tree

        if not self.root_layer_uid:
            tree.roots = gws.compact(enum(la.uid, []) for la in rd.project.map.layers)
            return tree

        uid = self.root_layer_uid
        if '.' not in uid:
            uid = rd.project.map.uid + '.' + uid
        root = enum(uid, [])
        if root:
            tree.roots.append(root)
        return tree

    SCOPE_LAYER = 1
    SCOPE_FEATURE = 2
    SCOPE_LEAF = 3

    def layer_caps_list(self, rd: Request, names: t.List[str] = None, scope: int = 0) -> t.List[LayerCaps]:
        """Return a list of leaf layer caps, optionally matching names."""

        tree = self.layer_caps_tree(rd)
        if names is None:
            return tree.leaves

        # the order is important, hence no sets
        name_list = gws.uniq(names)

        if scope == self.SCOPE_LAYER:
            return [
                lc
                for name in name_list
                for lc in tree.leaves
                if any(self._xname_matches(a.layer_xname, name) for a in lc.ancestors)
            ]

        if scope == self.SCOPE_FEATURE:
            return [
                lc
                for name in name_list
                for lc in tree.leaves
                if self._xname_matches(lc.feature_xname, name)
            ]

        if scope == self.SCOPE_LEAF:
            return [
                lc
                for name in name_list
                for lc in tree.leaves
                if self._xname_matches(lc.layer_xname, name)
            ]

    def layer_caps_list_from_request(self, rd: Request, param_names: t.List[str], scope: int, fallback_to_all=True) -> t.List[LayerCaps]:
        """Return a list of leaf layer caps matching request parameters."""

        names = None

        for p in param_names:
            if rd.req.has_param(p):
                names = gws.to_list(rd.req.param(p))
                break

        if names is None and fallback_to_all:
            return self.layer_caps_list(rd)

        return self.layer_caps_list(rd, names, scope)

    def _layer_options(self, layer: gws.ILayer, level: int) -> t.Optional[LayerOptions]:
        if not layer.ows_enabled:
            return None
        if self.is_raster_ows and not layer.supports_raster_ows:
            return None
        if self.is_vector_ows and not layer.supports_vector_ows:
            return None

        defaults = LayerOptions(
            layer_name=layer.uid.split('.')[-1],
            feature_name=layer.uid.split('.')[-1],
        )

        for lo in self.layer_options:
            if lo.level and lo.level != level:
                continue
            if lo.uids and layer.uid not in lo.uids:
                continue
            if lo.pattern and not re.search(lo.pattern, layer.uid):
                continue
            return gws.merge(defaults, lo) if lo.enabled else None

        return defaults

    _default_feature_name = 'feature'
    _default_geometry_name = 'geometry'

    def _populate_layer_caps(self, lc: LayerCaps, layer: gws.ILayer, lo: LayerOptions, children: t.List[LayerCaps]):
        lc.layer = layer
        lc.title = layer.title
        lc.layer_xname = self._parse_xname(lo.layer_name)
        lc.feature_xname = self._parse_xname(lo.feature_name)

        lc.meta = t.cast(gws.lib.metadata.Values, layer.metadata.values)
        lc.children = children

        lc.extent = layer.extent
        lc.extent4326 = gws.lib.extent.transform_to_4326(layer.extent, layer.crs)
        lc.has_legend = layer.has_legend or any(s.has_legend for s in lc.children)
        lc.has_search = layer.has_search or any(s.has_search for s in lc.children)

        scales = [gws.lib.units.res2scale(r) for r in layer.resolutions]
        lc.min_scale = int(min(scales))
        lc.max_scale = int(max(scales))

        lc.projections = [
            Projection(
                crs=crs,
                proj=gws.lib.proj.to_projection(crs),
                extent=gws.lib.extent.transform(layer.extent, layer.crs, crs)
            )
            for crs in self.supported_crs or [layer.crs]
        ]

        if not lc.feature_xname.ns.schema and layer.data_model:
            schema = layer.data_model.xml_schema(self._default_geometry_name)
            lc.adhoc_feature_schema = [
                FeatureSchemaAttribute(
                    type=typ,
                    xname=self._parse_xname(name, lc.feature_xname.ns.name))
                for name, typ in schema.items()
            ]

        return lc

    # FeatureCaps and collections

    def feature_collection(self, rd: Request, features: t.List[gws.IFeature], lcs: t.List[LayerCaps], populate=True, target_crs=None, invert_axis_if_geographic=False, crs_format='uri') -> FeatureCollection:
        coll = FeatureCollection(
            caps=[],
            features=[],
            time_stamp=gws.lib.date.now_iso(with_tz=False),
            num_matched=len(features),
            num_returned=len(features) if populate else 0,
        )

        if not populate:
            return coll

        layer_to_caps: t.Dict[str, LayerCaps] = {lc.layer.uid: lc for lc in lcs}

        default_xname = self._parse_xname(self._default_feature_name)
        prec = 2
        target_proj = None

        if target_crs:
            target_proj = gws.lib.proj.to_proj(target_crs)
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

            xname = default_xname
            if f.layer and f.layer.uid in layer_to_caps:
                xname = layer_to_caps[f.layer.uid].feature_xname

            coll.caps.append(FeatureCaps(
                feature=f,
                shape_tag=gs,
                xname=xname
            ))

            coll.features.append(f)

        return coll

    # Utils

    def service_url_path(self, project: t.Optional[gws.IProject] = None) -> str:
        return gws.action_url_path('owsService', serviceUid=self.uid, projectUid=project.uid if project else None)

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
        if out.items and out.items[0].image:
            content = out.items[0].image.to_bytes()
        else:
            content = gws.lib.image.PIXEL_PNG8

        return gws.ContentResponse(mime='image/png', content=content)

    def _parse_xname(self, name: str, nsid: str = None) -> XmlName:
        if ':' in name:
            nsid, name = name.split(':')

        if nsid:
            ns = self.xml_helper.namespace(nsid)
            if not ns:
                raise gws.Error(f'unknown namespace {nsid!r} for name {name!r}')
        else:
            ns = self.xml_helper.fallback_namespace

        return XmlName(
            p=name,
            q=ns.name + ':' + name,
            ns=ns)

    def _xname_matches(self, xname: XmlName, s: str) -> bool:
        if ':' in s:
            return xname.q == s
        return xname.p == s and xname.ns == self.xml_helper.fallback_namespace
