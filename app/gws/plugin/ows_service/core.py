import re

import gws
import gws.base.template
import gws.base.web.error
import gws.gis.crs
import gws.lib.date
import gws.gis.extent
import gws.gis.gml
import gws.lib.image
import gws.lib.metadata
import gws.lib.mime
import gws.gis.render
import gws.lib.uom as units
import gws.lib.xmlx as xmlx
import gws.types as t

_DEFAULT_NAMESPACE_PREFIX = 'gwsns'
_DEFAULT_NAMESPACE_URI = 'http://gbd-websuite.de'
_DEFAULT_FEATURE_NAME = 'feature'
_DEFAULT_GEOMETRY_NAME = 'geometry'


class Error(gws.Error):
    pass


class LayerFilter(gws.Data):
    """Layer filter"""

    level: int = 0 
    """match only layers at this level"""
    uids: t.Optional[list[str]] 
    """match these layer uids"""
    pattern: gws.Regex = '' 
    """match layers whose uid matches a pattern"""


class LayerConfig(gws.Config):
    """Layer-specific OWS configuration"""

    applyTo: t.Optional[LayerFilter] 
    """project layers this configuration applies to"""
    enabled: bool = True 
    """layer is enabled for this service"""
    layerName: t.Optional[str] 
    """layer name for this service"""
    layerTitle: t.Optional[str] 
    """layer title for this service"""
    featureName: t.Optional[str] 
    """feature name for this service"""


class ServiceConfig(gws.ConfigWithAccess):
    layerConfig: t.Optional[list[LayerConfig]] 
    """custom configurations for specific layers"""
    metadata: t.Optional[gws.Metadata] 
    """service metadata"""
    rootLayer: str = '' 
    """root layer uid"""
    strictParams: bool = False 
    """strict parameter parsing"""
    supportedCrs: t.Optional[list[gws.CrsName]] 
    """supported CRS for this service"""
    templates: t.Optional[list[gws.ext.config.template]] 
    """service XML templates"""
    updateSequence: t.Optional[str] 
    """service update sequence"""
    withInspireMeta: bool = False 
    """use INSPIRE Metadata"""


##


class Request(gws.Data):
    req: gws.IWebRequester
    project: gws.IProject
    service: gws.IOwsService
    xml_element: t.Optional[gws.IXmlElement] = None
    xml_is_soap: bool = False


FeatureSchemaAttribute = tuple[str, str]  # type, name


class LayerOptions(gws.Data):
    level: int
    uids: set[str]
    pattern: str
    enabled: bool
    layer_name: str
    layer_title: str
    feature_name: str


class LayerCaps(gws.Data):
    layer: gws.ILayer

    has_legend: bool
    has_search: bool
    meta: gws.Metadata
    title: str

    layer_pname: str
    layer_qname: str
    feature_pname: str
    feature_qname: str

    extent: gws.Extent
    wgsExtent: gws.Extent
    max_scale: int
    min_scale: int
    bounds: list[gws.Bounds]

    children: list['LayerCaps']
    ancestors: list['LayerCaps']

    adhoc_feature_schema: t.Optional[dict]


class LayerCapsTree(gws.Data):
    roots: list[LayerCaps]
    leaves: list['LayerCaps']


class FeatureCaps(gws.Data):
    feature: gws.IFeature
    shape_element: gws.IXmlElement
    qname: str


class FeatureCollection(gws.Data):
    caps: list[FeatureCaps]
    features: list[gws.IFeature]
    time_stamp: str
    num_matched: int
    num_returned: int


##


class Service(gws.Node, gws.IOwsService):
    """Baseclass for OWS services."""

    project: t.Optional[gws.IProject]

    root_layer_uid: str
    update_sequence: str
    layer_options: list[LayerOptions]

    is_raster_ows: bool = False
    is_vector_ows: bool = False

    supported_crs: list[gws.ICrs]

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

        self.root_layer_uid = self.cfg('rootLayer')
        self.supported_crs = [gws.gis.crs.require(s) for s in self.cfg('supportedCrs', default=[])]
        self.update_sequence = self.cfg('updateSequence')
        self.with_inspire_meta = self.cfg('withInspireMeta')
        self.with_strict_params = self.cfg('withStrictParams')

        self.templates = gws.base.template.manager.create(
            self.root,
            items=self.cfg('templates'),
            defaults=self.default_templates,
            parent=self)

        self.layer_options = []

        for cfg in self.cfg('layerConfig', default=[]):
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
        m = gws.lib.metadata.from_config(self.cfg('metadata'))
        m.extend(
            self.project.metadata if self.project else self.root.app.metadata,
            self.default_metadata,
            {'name': self.protocol})

        if not m.get('extraLinks') and self.service_link:
            m.set('extraLinks', [self.service_link])

        return m

    def activate(self):
        xmlx.namespaces.add(_DEFAULT_NAMESPACE_PREFIX, _DEFAULT_NAMESPACE_URI)

    # Request handling

    def handle_request(self, req: gws.IWebRequester) -> gws.ContentResponse:
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

        return self.dispatch_request(rd, req.param('request', ''))

    def dispatch_request(self, rd: Request, request_param):
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

    def template_response(self, rd: Request, verb: gws.OwsVerb, format: str = None, context=None):
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

        return tpl.render(gws.TemplateRenderInput(context=context))

    def enum_template_formats(self):
        fs = {}

        for tpl in self.templates.items:
            for mime in tpl.mimes:
                fs.setdefault(tpl.name, set())
                fs[tpl.name].add(mime)

        return {k: sorted(v) for k, v in fs.items()}

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

    def layer_caps_list(self, rd: Request, names: list[str] = None, scope: int = 0) -> list[LayerCaps]:
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
                if any(self._qname_matches(a.layer_qname, name) for a in lc.ancestors)
            ]

        if scope == self.SCOPE_FEATURE:
            return [
                lc
                for name in name_list
                for lc in tree.leaves
                if self._qname_matches(lc.feature_qname, name)
            ]

        if scope == self.SCOPE_LEAF:
            return [
                lc
                for name in name_list
                for lc in tree.leaves
                if self._qname_matches(lc.layer_qname, name)
            ]

    def _qname_matches(self, qname: str, s: str) -> bool:
        if ':' in s:
            return qname == s
        pfx, n = qname.split(':')
        return n == s and pfx == _DEFAULT_NAMESPACE_PREFIX

    def layer_caps_list_from_request(self, rd: Request, param_names: list[str], scope: int, fallback_to_all=True) -> list[LayerCaps]:
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

    def _populate_layer_caps(self, lc: LayerCaps, layer: gws.ILayer, lo: LayerOptions, children: list[LayerCaps]):
        lc.layer = layer
        lc.title = layer.title

        lc.layer_qname = xmlx.qualify_name(lo.layer_name, _DEFAULT_NAMESPACE_PREFIX)
        lc.layer_pname = xmlx.unqualify_name(lc.layer_qname)

        lc.feature_qname = xmlx.qualify_name(lo.feature_name, _DEFAULT_NAMESPACE_PREFIX)
        lc.feature_pname = xmlx.unqualify_name(lc.feature_qname)

        lc.meta = layer.metadata.values
        lc.children = children

        lc.extent = layer.extent
        lc.wgsExtent = gws.gis.extent.transform_to_4326(layer.extent, layer.crs)
        lc.has_legend = layer.has_legend or any(s.has_legend for s in lc.children)
        lc.has_search = layer.has_search or any(s.has_search for s in lc.children)

        scales = [gws.lib.uom.res_to_scale(r) for r in layer.resolutions]
        lc.min_scale = int(min(scales))
        lc.max_scale = int(max(scales))

        lc.bounds = [
            gws.Bounds(
                crs=crs,
                extent=gws.gis.extent.transform(layer.extent, layer.crs, crs)
            )
            for crs in self.supported_crs or [layer.crs]
        ]

        pfx, n = xmlx.split_name(lc.feature_qname)

        if not xmlx.namespaces.schema(pfx) and layer.data_model:
            lc.adhoc_feature_schema = layer.data_model.xml_schema_dict(name_for_geometry=_DEFAULT_GEOMETRY_NAME)

        return lc

    # FeatureCaps and collections

    def feature_collection(
        self,
        rd: Request,
        features: list[gws.IFeature],
        lcs: list[LayerCaps],
        target_crs: gws.ICrs,
        populate=True,
        invert_axis_if_geographic=False,
        crs_format: gws.CrsFormat = gws.CrsFormat.URI
    ) -> FeatureCollection:
        ##
        coll = FeatureCollection(
            caps=[],
            features=[],
            time_stamp=gws.lib.date.now_iso(with_tz=False),
            num_matched=len(features),
            num_returned=len(features) if populate else 0,
        )

        if not populate:
            return coll

        layer_to_caps: dict[str, LayerCaps] = {lc.layer.uid: lc for lc in lcs}

        default_qname = xmlx.qualify_name(_DEFAULT_FEATURE_NAME, _DEFAULT_NAMESPACE_PREFIX)

        axis = gws.AXIS_XY
        if target_crs.is_geographic and invert_axis_if_geographic:
            axis = gws.AXIS_YX

        precision = 6 if target_crs.is_geographic else 2

        for f in features:
            f.transform_to(target_crs)

            shape_element = None
            if f.shape:
                shape_element = gws.gis.gml.shape_to_element(
                    f.shape, precision, axis, crs_format, with_ns='gml')

            f.apply_data_model()

            qname = default_qname
            if f.layer and f.layer.uid in layer_to_caps:
                qname = layer_to_caps[f.layer.uid].feature_qname

            coll.caps.append(FeatureCaps(
                feature=f,
                shape_element=shape_element,
                qname=qname
            ))

            coll.features.append(f)

        return coll

    # Utils

    def service_url_path(self, project: t.Optional[gws.IProject] = None) -> str:
        return gws.action_url_path('owsService', serviceUid=self.uid, projectUid=project.uid if project else None)

    def render_map_bbox_from_layer_caps_list(self, rd: Request, lcs: list[LayerCaps], bounds: gws.Bounds) -> gws.ContentResponse:
        try:
            px_width = int(rd.req.param('width'))
            px_height = int(rd.req.param('height'))
        except:
            raise gws.base.web.error.BadRequest()

        if not bounds or not px_width or not px_height:
            raise gws.base.web.error.BadRequest()

        mri = gws.MapRenderInput(
            background_color=0 if rd.req.param('transparent', '').lower() == 'false' else None,
            bbox=bounds.extent,
            crs=bounds.crs,
            out_size=(px_width, px_height, gws.Uom.px),
            planes=[
                gws.MapRenderInputPlane(type='image_layer', layer=lc.layer)
                for lc in lcs
            ]
        )

        mro = gws.gis.render.render_map(mri)

        if mro.planes and mro.planes[0].image:
            content = mro.planes[0].image.to_bytes()
        else:
            content = gws.lib.image.PIXEL_PNG8

        return gws.ContentResponse(mime=gws.lib.mime.PNG, content=content)
