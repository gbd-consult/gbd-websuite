import gws
import gws.web.error
import gws.tools.misc
import gws.tools.xml3
import gws.gis.proj
import gws.ows.gml
import gws.common.datamodel
import gws.common.search.runner
import gws.types as t
import gws.web

NAMESPACES = {
    'csw': "http://www.opengis.net/cat/csw/2.0.2",
    'dc': "http://purl.org/dc/elements/1.1/",
    'dct': "http://purl.org/dc/terms/",
    'fes': "http://www.opengis.net/fes/2.0",
    'gmd': "http://www.isotc211.org/2005/gmd",
    'gml': "http://www.opengis.net/gml/3.2",
    'ogc': "http://www.opengis.net/ogc",
    'ows': "http://www.opengis.net/ows/1.1",
    'rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns",
    'sld': "http://www.opengis.net/sld",
    'wfs': "http://www.opengis.net/wfs/2.0",
    'wms': "http://www.opengis.net/wms",
    'xlink': "http://www.w3.org/1999/xlink",
    'xsd': "http://www.w3.org/2001/XMLSchema",
    'xsi': "http://www.w3.org/2001/XMLSchema-instance",
}

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


class Config(t.WithTypeAndAccess):
    enabled: bool = True
    featureNamespace: str = 'gws'  #: feature namespace name
    featureNamespaceUri: str = 'https://gws.gbd-consult.de'  #: feature namespace uri


class RequestData(t.Data):
    req: gws.web.AuthRequest
    project: t.ProjectObject
    service: 'Object'
    xml: gws.tools.xml3.Element = None


class LayerCapsNode(t.Data):
    layer: t.LayerObject
    extent: t.Extent
    lonlat_extent: t.Extent
    proj: gws.gis.proj.Proj
    has_search: bool
    min_scale: int
    max_scale: int
    sub_nodes: t.Optional[t.List['LayerCapsNode']]
    feature_schema: t.List[t.Attribute]


class FeatureNode(t.Data):
    feature: t.FeatureInterface
    shape_tag: ''
    tag_name: ''
    attributes: t.List[t.Attribute]


class Object(gws.Object):
    """Generic OWS Service."""

    def __init__(self):
        super().__init__()
        self.base_path = ''
        self.feature_namespace = ''
        self.namespaces = gws.extend({}, NAMESPACES)
        self.service_type = ''
        self.templates = {}
        self.version = ''

    def configure(self):
        super().configure()

        self.service_type = self.var('type')

        if self.var('featureNamespace'):
            self.feature_namespace = self.var('featureNamespace')
            self.namespaces[self.feature_namespace] = self.var('featureNamespaceUri')

    def can_handle(self, req) -> bool:
        return req.kparam('service', '').lower() == self.service_type

    def error_response(self, status):
        return xml_error_response(self.version, status, f'Error {status}')

    def handle(self, req) -> t.HttpResponse:
        rd = RequestData({
            'req': req,
            'project': req.require_project(req.param('projectUid')),
            'service': self,
        })
        return self.dispatch(rd, req.kparam('request', '').lower())

    def dispatch(self, rd: RequestData, request_param):
        h = getattr(self, 'handle_' + request_param, None)
        if not h:
            gws.log.debug(f'request={request_param!r} not found')
            raise gws.web.error.NotFound()
        return h(rd)

    def configure_template(self, name, type='xml'):
        p = self.var('templates.' + name)
        if p:
            return self.create_object('gws.ext.template', p)

        name = name.replace('.', '/')
        return self.create_shared_object('gws.ext.template', self.base_path + name, {
            'type': type,
            'path': self.base_path + f'/templates/{name}.cx'
        })

    def is_layer_enabled(self, layer):
        return layer and layer.has_ows(self.service_type)

    def service_endpoint(self, rd: RequestData):
        u = '/_/cmd/owsHttp'
        if rd.project:
            u += f'/projectUid/{rd.project.uid}'
        return u

    def render_template(self, rd: RequestData, template, context, format=None):
        context = gws.defaults(context, {
            'project': rd.project,
            'feature_namespace': self.feature_namespace,
            'namespaces': self.namespaces,
            'service_version': self.version,
            'service_endpoint': self.service_endpoint(rd),
        })
        return self.templates[template].render(context, format=format).content


def layer_node_tree(rd: RequestData) -> t.List[LayerCapsNode]:
    return gws.compact(_layer_node_subtree(rd, la.uid) for la in rd.project.map.layers)


def layer_node_list(rd: RequestData) -> t.List[LayerCapsNode]:
    all_nodes = []
    for node in layer_node_tree(rd):
        _layer_node_sublist(rd, node, all_nodes)
    return all_nodes


def layer_nodes_from_request_params(rd: RequestData, *params):
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

        if p[0] == rd.service.feature_namespace:
            ows_names.add(s)
            ows_names.add(p[1])
        else:
            ows_names.add(s)

    nodes = layer_node_list(rd)
    nodes = [n for n in nodes if n.layer.ows_name in ows_names]

    return nodes


def feature_node_list(rd: RequestData, features):
    return [_feature_node(rd, f) for f in features]


def render_feature_nodes(rd: RequestData, nodes, container_template_name):
    tags = []
    used_namespaces = set()

    for node in nodes:
        template_name = node.get('template_name') or 'feature'
        ns, tag = rd.service.render_template(rd, template_name, {'node': node}, format='tag')
        used_namespaces.update(ns)
        tags.append(tag)

    return xml_response(rd.service.render_template(rd, container_template_name, {
        'feature_tags': tags,
        'used_namespaces': used_namespaces,
    }))


def xml_error_response(version, status, description):
    description = gws.tools.xml3.encode(description)
    content = f'<?xml version="1.0" encoding="UTF-8"?>' \
              f'<ServiceExceptionReport version="{version}">' \
              f'<ServiceException code="{status}">{description}</ServiceException>' \
              f'</ServiceExceptionReport>'
    return xml_response(content, status)


def xml_response(content, status=200):
    return t.HttpResponse({
        'mimeType': 'text/xml',
        'content': gws.tools.xml3.as_string(content),
        'status': status,
    })


##

def _layer_node_subtree(rd: RequestData, layer_uid):
    layer = rd.req.acquire('gws.ext.layer', layer_uid)

    if not rd.service.is_layer_enabled(layer):
        return

    if not layer.layers:
        return _layer_node(rd, layer)

    sub = gws.compact(_layer_node_subtree(rd, la.uid) for la in layer.layers)
    if sub:
        return _layer_node(rd, layer, sub)


def _layer_node_sublist(rd: RequestData, node, all_nodes):
    if not node.sub_nodes:
        all_nodes.append(node)
        return
    for n in node.sub_nodes:
        _layer_node_sublist(rd, n, all_nodes)


def _layer_node(rd: RequestData, layer, sub_nodes=None) -> LayerCapsNode:
    res = [gws.tools.misc.res2scale(r) for r in layer.resolutions]
    crs = layer.map.crs
    sub_nodes = sub_nodes or []

    return LayerCapsNode({
        'layer': layer,
        'tag_name': layer.ows_name,
        'extent': layer.extent,
        'lonlat_extent': ['%.3f1' % c for c in gws.gis.proj.transform_bbox(layer.extent, crs, 'EPSG:4326')],
        'proj': gws.gis.proj.as_proj(crs),
        'has_search': layer.has_search or any(s['has_search'] for s in sub_nodes),
        'min_scale': min(res),
        'max_scale': max(res),
        'sub_nodes': sub_nodes,
    })


def _feature_node(rd: RequestData, feature: t.FeatureInterface):
    gs = None
    if feature.shape:
        gs = gws.ows.gml.shape_to_tag(feature.shape, precision=rd.project.map.coordinate_precision)

    atts = feature.attributes or {}

    la = feature.layer
    dm = la.data_model if la else None
    atts = gws.common.datamodel.apply(dm, atts)
    name = la.ows_name if la else 'feature'

    return FeatureNode({
        'feature': feature,
        'shape_tag': gs,
        'tag_name': name,
        'attributes': atts
    })
