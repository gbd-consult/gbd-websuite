import gws
import gws.common.metadata
import gws.common.model
import gws.common.search.runner
import gws.gis.gml
import gws.gis.proj
import gws.tools.units as units
import gws.tools.xml3
import gws.types as t
import gws.web.error

from . import const, inspire

# NB our templates use "xsd" for the XMLSchema namespace

ATTR_TYPE_TO_XML = {
    t.AttributeType.bool: 'xsd:boolean',
    t.AttributeType.bytes: None,
    t.AttributeType.date: 'xsd:date',
    t.AttributeType.datetime: 'datetime',
    t.AttributeType.float: 'xsd:decimal',
    t.AttributeType.geometry: None,
    t.AttributeType.int: 'xsd:integer',
    t.AttributeType.list: None,
    t.AttributeType.str: 'xsd:string',
    t.AttributeType.time: 'xsd:time',
    t.AttributeType.geoCurve: 'gml:CurvePropertyType',
    t.AttributeType.geoGeomcollection: 'gml:MultiGeometryPropertyType',
    t.AttributeType.geoGeometry: 'gml:MultiGeometryPropertyType',
    t.AttributeType.geoLinestring: 'gml:CurvePropertyType',
    t.AttributeType.geoMulticurve: 'gml:MultiCurvePropertyType',
    t.AttributeType.geoMultilinestring: 'gml:MultiCurvePropertyType',
    t.AttributeType.geoMultipoint: 'gml:MultiPointPropertyType',
    t.AttributeType.geoMultipolygon: 'gml:MultiGeometryPropertyType',
    t.AttributeType.geoMultisurface: 'gml:MultiGeometryPropertyType',
    t.AttributeType.geoPoint: 'gml:PointPropertyType',
    t.AttributeType.geoPolygon: 'gml:SurfacePropertyType',
    t.AttributeType.geoPolyhedralsurface: 'gml:SurfacePropertyType',
    t.AttributeType.geoSurface: 'gml:SurfacePropertyType',
}


class UseInspireConfig(t.Config):
    meta: bool = False  #: enable INSPIRE metadata for this service
    data: bool = False  #: enable INSPIRE data models for this service


class Config(t.WithTypeAndAccess):
    enabled: bool = True
    featureNamespace: str = 'gws'  #: feature namespace name
    featureNamespaceUri: str = 'http://gws.gbd-consult.de'  #: feature namespace uri
    meta: t.Optional[gws.common.metadata.Config]  #: service metadata
    name: str = ''  #: service name
    templates: t.Optional[t.List[t.ext.template.Config]]  #: service XML templates
    useInspire: t.Optional[UseInspireConfig]  #: INSPIRE configuration


class OwsRequest(t.Data):
    req: t.IRequest
    project: t.IProject
    service: t.IOwsService
    xml: gws.tools.xml3.Element = None


class LayerCapsNode(t.Data):
    layer: t.ILayer = None

    feature_schema: t.List[t.Attribute] = []
    has_legend: bool = False
    has_search: bool = False
    meta: t.MetaData = None
    tag_name: str = None
    title: str = None

    extent: t.Extent = None
    lonlat_extent: t.Extent = None
    max_scale: int = None
    min_scale: int = None
    proj: gws.gis.proj.Proj = None

    sub_nodes: t.Optional[t.List['LayerCapsNode']] = []


class FeatureNode(t.Data):
    feature: t.IFeature
    shape_tag: ''
    tag_name: ''
    attributes: t.List[t.Attribute]


_NAMESPACES = gws.extend({}, const.NAMESPACES, inspire.NAMESPACES)


#:export IOwsService
class Object(gws.Object, t.IOwsService):
    """OWS service interface."""

    def __init__(self):
        super().__init__()

        self.feature_namespace = ''
        self.meta: t.MetaData = None
        self.name = ''
        self.type = ''
        self.version = ''
        self.enabled = True

    def handle(self, req: t.IRequest) -> t.HttpResponse:
        pass

    def error_response(self, status) -> t.HttpResponse:
        pass


class Base(Object):
    """Baseclass for OWS services."""

    def __init__(self):
        super().__init__()

        self.namespaces = _NAMESPACES
        self.local_namespaces = {}
        self.templates = {}
        self.use_inspire_meta = False
        self.use_inspire_data = False

        self.project: t.IProject = None

    @property
    def url(self):
        u = gws.SERVER_ENDPOINT + '/cmd/owsHttp/serviceName/' + self.name
        if self.project:
            u += f'/projectUid/{self.project.uid}'
        return u

    @property
    def service_link(self):
        return None

    ## Configuration

    def configure(self):
        super().configure()

        self.project: t.IProject = self.get_closest('gws.common.project')
        self.name = self.var('name') or self.type

        m = self._metadata_dict(self.var('meta'))
        if self.project:
            # use project metadata as a fallback
            m = gws.extend(self._metadata_dict(self.project.meta), m)
        self.meta = gws.common.metadata.read(m)

        if self.var('featureNamespace'):
            self.feature_namespace = self.var('featureNamespace')
            self.local_namespaces[self.feature_namespace] = self.var('featureNamespaceUri')

        self.use_inspire_meta = self.var('useInspire.meta')
        self.use_inspire_data = self.var('useInspire.data')

        if self.use_inspire_data:
            self.configure_inspire_templates()

    def _metadata_dict(self, meta):
        m = gws.extend({}, meta)

        m['inspire'] = gws.extend({
            'mandatoryKeyword': 'infoMapAccessService',
            'resourceType': 'service',
            'spatialDataServiceType': 'view'
        }, m.get('inspire', {}))

        m['iso'] = gws.extend({
            'scope': 'service'
        }, m.get('iso', {}))

        m['serviceUrl'] = m.get('serviceUrl', self.url)

        ls = []

        if self.service_link:
            ls.append(self.service_link)

        m['links'] = ls + m.get('links', [])

        return gws.strip(m)

    def configure_template(self, name, path, type='xml'):
        for tpl in self.var('templates', default=[]):
            if tpl.title == name:
                return self.create_object('gws.ext.template', tpl)

        if not path.startswith('/'):
            path = gws.APP_DIR + '/gws/ext/ows/service/' + path.strip('/') + '/' + name + '.cx'

        return self.create_shared_object('gws.ext.template', path, {
            'type': type,
            'path': path
        })

    def configure_inspire_templates(self):
        for tag in inspire.TAGS:
            self.templates[tag] = self.configure_template(tag.replace(':', '_'), 'wfs/templates/inspire')

    ## Request handling

    def handle(self, req) -> t.HttpResponse:
        rd = OwsRequest({
            'req': req,
            'project': req.require_project(req.param('projectUid')),
        })
        return self.dispatch(rd, req.kparam('request', '').lower())

    def dispatch(self, rd: OwsRequest, request_param):
        h = getattr(self, 'handle_' + request_param, None)
        if not h:
            gws.log.debug(f'request={request_param!r} not found')
            raise gws.web.error.NotFound()
        return h(rd)

    ## Rendering and responses

    def error_response(self, status):
        return self.xml_error_response(self.version, status, f'Error {status}')

    def render_template(self, rd: OwsRequest, template, context, format=None):

        def csw_meta_url(uid):
            return rd.req.url_for(
                gws.SERVER_ENDPOINT + '/cmd/owsHttp/serviceName/csw/request/GetRecordById/id/' + gws.as_uid(uid))

        context = gws.extend({
            'project': rd.project,
            'meta': self.meta,
            'use_inspire_meta': self.use_inspire_meta,
            'url_for': rd.req.url_for,
            'csw_meta_url': csw_meta_url,
            'feature_namespace': self.feature_namespace,
            'feature_namespace_uri': self.local_namespaces[self.feature_namespace],
            'all_namespaces': self.namespaces,
            'local_namespaces': self.local_namespaces,
            'service': self
        }, context)

        return self.templates[template].render(context, format=format).content

    def render_feature_nodes(self, rd: OwsRequest, nodes: t.List[FeatureNode], container_template_name: str) -> t.HttpResponse:
        tags = []

        for node in nodes:
            template_name = node.tag_name if node.tag_name in self.templates else 'feature'
            tags.append(self.render_template(rd, template_name, {'node': node}, format='tag'))

        return self.xml_response(self.render_template(rd, container_template_name, {
            'feature_tags': tags,
        }))

    def xml_error_response(self, version, status, description) -> t.HttpResponse:
        description = gws.tools.xml3.encode(description)
        content = (f'<?xml version="1.0" encoding="UTF-8"?>'
                   + f'<ServiceExceptionReport version="{version}">'
                   + f'<ServiceException code="{status}">{description}</ServiceException>'
                   + f'</ServiceExceptionReport>')
        return self.xml_response(content, status)

    def xml_response(self, content, status=200) -> t.HttpResponse:
        return t.HttpResponse({
            'mime': 'text/xml',
            'content': gws.tools.xml3.as_string(content),
            'status': status,
        })

    ## LayerCaps nodes

    def layer_node_root(self, rd: OwsRequest) -> t.Optional[LayerCapsNode]:
        """Return a single root node for a layer tree (for WMS)."""

        roots = self._layer_node_roots(rd)

        if not roots:
            return

        if len(roots) == 1:
            return roots[0]

        # multiple root layers -> create a root node from the project

        p = rd.project

        root = LayerCapsNode(
            meta=p.meta,
            sub_nodes=roots,
            tag_name=p.uid,
            title=p.title,
        )

        return self._add_spatial_props(root, p.map.extent, p.map.crs, p.map.resolutions)

    def layer_node_list(self, rd: OwsRequest) -> t.List[LayerCapsNode]:
        """Return a list of terminal layer nodes (for WFS)."""

        all_nodes = []
        for node in self._layer_node_roots(rd):
            self._layer_node_sublist(rd, node, all_nodes)
        return all_nodes

    def layer_nodes_from_request_params(self, rd: OwsRequest, *params):
        ls = []
        for p in params:
            ls = ls or gws.as_list(rd.req.kparam(p))
        ows_names = set()

        for s in ls:
            if ':' not in s:
                ows_names.add(s)
                continue

            p = s.split(':')
            if len(p) != 2:
                continue

            if p[0] == self.feature_namespace:
                ows_names.add(s)
                ows_names.add(p[1])
            else:
                ows_names.add(s)

        all_nodes = []
        for node in self._layer_node_roots(rd):
            if node.tag_name in ows_names:
                self._layer_node_sublist(rd, node, all_nodes)
            else:
                self._layer_node_sublist_selected(rd, node, all_nodes, ows_names)

        return all_nodes

    def _layer_node_roots(self, rd: OwsRequest) -> t.List[LayerCapsNode]:
        if not rd.project:
            return []
        return gws.compact(self._layer_node_subtree(rd, la.uid) for la in rd.project.map.layers)

    def _layer_node_subtree(self, rd: OwsRequest, layer_uid):
        layer: t.ILayer = rd.req.acquire('gws.ext.layer', layer_uid)
        if not self.is_layer_enabled(layer):
            return

        if not layer.layers:
            return self._layer_node_from(layer)

        sub = gws.compact(self._layer_node_subtree(rd, la.uid) for la in layer.layers)
        if sub:
            return self._layer_node_from(layer, sub)

    def _layer_node_sublist(self, rd: OwsRequest, node: LayerCapsNode, all_nodes):
        if not node.sub_nodes:
            all_nodes.append(node)
            return
        for n in node.sub_nodes:
            self._layer_node_sublist(rd, n, all_nodes)

    def _layer_node_sublist_selected(self, rd: OwsRequest, node: LayerCapsNode, all_nodes, tag_names):
        if node.tag_name in tag_names:
            if not node.sub_nodes:
                all_nodes.append(node)
                return
            for n in node.sub_nodes:
                self._layer_node_sublist(rd, n, all_nodes)
        elif node.sub_nodes:
            for n in node.sub_nodes:
                self._layer_node_sublist_selected(rd, n, all_nodes, tag_names)

    def _layer_node_from(self, layer: t.ILayer, sub_nodes=None) -> LayerCapsNode:
        sub_nodes = sub_nodes or []

        node = LayerCapsNode(
            has_search=layer.has_search or any(n.has_search for n in sub_nodes),
            has_legend=layer.has_legend,
            layer=layer,
            meta=layer.meta,
            sub_nodes=sub_nodes,
            tag_name=layer.ows_name,
            title=layer.title,
        )

        return self._add_spatial_props(node, layer.extent, layer.map.crs, layer.resolutions)

    def _add_spatial_props(self, node: LayerCapsNode, extent: t.Extent, crs, resolutions):
        scales = [units.res2scale(r) for r in resolutions]
        node.extent = extent
        node.lonlat_extent = self.lonlat_extent(extent, crs)
        node.max_scale = max(scales)
        node.min_scale = min(scales)
        node.proj = gws.gis.proj.as_proj(crs)
        return node

    def inspire_nodes(self, nodes):
        return [n for n in nodes if n.tag_name in inspire.TAGS]

    def feature_node_list(self, rd: OwsRequest, features: t.List[t.IFeature]):
        def node(f: t.IFeature):
            gs = None
            if f.shape:
                gs = gws.gis.gml.shape_to_tag(f.shape, precision=rd.project.map.coordinate_precision)

            f.convert()

            return FeatureNode({
                'feature': f,
                'shape_tag': gs,
                'tag_name': f.layer.ows_name if f.layer else 'feature',
                'attributes': f.attributes,
            })

        return [node(f) for f in features]

    def collect_iso_metadata(self, service):
        rs = {}

        for obj in service.find_all():
            meta = getattr(obj, 'meta', None)
            uid = gws.get(meta, 'iso.uid')
            if not uid:
                continue
            m = self._configure_metadata(obj)
            if m:
                rs[gws.as_uid(uid)] = m
        return rs

    ##

    def _configure_metadata(self, obj: gws.Object):
        m: t.MetaData = gws.common.metadata.read(obj.meta)

        if obj.is_a('gws.common.project'):
            la = obj.map
        elif obj.is_a('gws.ext.layer'):
            la = obj
        else:
            la = obj.get_closest('gws.common.project')
            if la:
                la = la.map

        if la:
            m.proj = gws.gis.proj.as_proj(la.crs)
            m.lonlat_extent = lonlat_extent(la.extent, la.crs)
            m.resolution = int(min(units.res2scale(r) for r in la.resolutions))

        if gws.get(m, 'inspire.theme'):
            m.inspire['themeName'] = inspire.theme_name(m.inspire['theme'], m.language)

        m.iso = gws.extend({
            'spatialType': 'vector',
        }, m.iso)

        m.inspire = gws.extend({
            'qualityExplanation': '',
            'qualityPass': 'false',
            'qualityLineage': '',
        }, m.inspire)

        return m

    ## Utils

    def is_layer_enabled(self, layer):
        return layer and layer.ows_enabled(self)

    def lonlat_extent(self, extent, crs):
        return [round(c, 4) for c in gws.gis.proj.transform_extent(extent, crs, 'EPSG:4326')]
