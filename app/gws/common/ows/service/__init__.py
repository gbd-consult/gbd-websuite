import gws
import gws.web.error
import gws.tools.misc
import gws.tools.xml3
import gws.gis.proj
import gws.ows.gml
import gws.common.datamodel
import gws.common.metadata
import gws.common.search.runner
import gws.types as t
import gws.web

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


class Config(t.WithTypeAndAccess):
    enabled: bool = True
    featureNamespace: str = 'gws'  #: feature namespace name
    featureNamespaceUri: str = 'http://gws.gbd-consult.de'  #: feature namespace uri
    meta: t.Optional[t.MetaData]  #: service metadata


class RequestData(t.Data):
    req: gws.web.AuthRequest
    project: t.ProjectObject
    service: 'Object'
    xml: gws.tools.xml3.Element = None


class LayerCapsNode(t.Data):
    layer: t.LayerObject

    title: str
    tag_name: str
    has_search: bool
    meta: t.MetaData
    feature_schema: t.List[t.Attribute]

    extent: t.Extent
    lonlat_extent: t.Extent
    max_scale: int
    min_scale: int
    proj: gws.gis.proj.Proj

    sub_nodes: t.Optional[t.List['LayerCapsNode']]


class FeatureNode(t.Data):
    feature: t.FeatureInterface
    shape_tag: ''
    tag_name: ''
    attributes: t.List[t.Attribute]


_NAMESPACES = gws.extend({}, const.NAMESPACES, inspire.NAMESPACES)


class Object(gws.Object):
    """Generic OWS Service."""

    def __init__(self):
        super().__init__()
        self.feature_namespace = ''
        self.meta = None
        self.namespaces = _NAMESPACES
        self.local_namespaces = {}
        self.service_class = ''
        self.service_type = ''
        self.templates = {}
        self.version = ''

    def configure(self):
        super().configure()

        if self.var('meta'):
            self.meta = gws.common.metadata.read(self.var('meta'))

            self.meta.inspire = gws.extend({
                'mandatoryKeyword': 'infoMapAccessService',
                'resourceType': 'service',
                'spatialDataServiceType': 'view'
            }, self.meta.inspire)

            self.meta.iso = gws.extend({
                'scope': 'service'
            }, self.meta.iso)

        if self.var('featureNamespace'):
            self.feature_namespace = self.var('featureNamespace')
            self.local_namespaces[self.feature_namespace] = self.var('featureNamespaceUri')

    def can_handle(self, req) -> bool:
        return req.kparam('srv', '').lower() == self.service_class

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

    def configure_template(self, name, path, type='xml'):
        p = self.var('templates.' + name)
        if p:
            return self.create_object('gws.ext.template', p)

        if not path.startswith('/'):
            path = gws.APP_DIR + '/gws/ext/ows/service/' + path.strip('/') + '/' + name + '.cx'

        return self.create_shared_object('gws.ext.template', path, {
            'type': type,
            'path': path
        })

    def is_layer_enabled(self, layer):
        return layer and layer.has_ows(self.service_type)

    def service_endpoint(self, rd: RequestData):
        u = gws.SERVER_ENDPOINT + '/cmd/owsHttp/srv/' + self.service_class
        if rd.project:
            u += f'/projectUid/{rd.project.uid}'
        return u

    def render_template(self, rd: RequestData, template, context, format=None):

        def rewrite_url(url, *parts):
            if parts:
                url = url.rstrip('/') + '/' + '/'.join(gws.as_uid(p) for p in parts)
            return rd.req.rewritten_url(url)

        context = gws.extend({
            'project': rd.project,
            'meta': self.meta or (rd.project.meta if rd.project else {}),
            'rewrite_url': rewrite_url,
            'feature_namespace': self.feature_namespace,
            'feature_namespace_uri': self.local_namespaces[self.feature_namespace],
            'all_namespaces': self.namespaces,
            'local_namespaces': self.local_namespaces,
            'service': {
                'version': self.version,
                'endpoint': self.service_endpoint(rd),
            }
        }, context)
        return self.templates[template].render(context, format=format).content


def layer_node_root(rd: RequestData) -> t.List[LayerCapsNode]:
    roots = _layer_node_roots(rd)
    if not roots:
        return
    if len(roots) == 1:
        return roots[0]
    # @TODO create root node from the project
    return roots[0]


def layer_node_list(rd: RequestData) -> t.List[LayerCapsNode]:
    all_nodes = []
    for node in _layer_node_roots(rd):
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
    nodes = [n for n in nodes if n.tag_name in ows_names]

    return nodes


def feature_node_list(rd: RequestData, features):
    return [_feature_node(rd, f) for f in features]


def render_feature_nodes(rd: RequestData, nodes, container_template_name):
    tags = []

    for node in nodes:
        template_name = node.get('template_name') or 'feature'
        tags.append(rd.service.render_template(rd, template_name, {'node': node}, format='tag'))

    return xml_response(rd.service.render_template(rd, container_template_name, {
        'feature_tags': tags,
    }))


def lonlat_extent(extent, crs):
    return ['%.3f1' % c for c in gws.gis.proj.transform_bbox(extent, crs, 'EPSG:4326')]


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


def collect_metadata(service):
    rs = {}

    for obj in service.find_all():
        meta = getattr(obj, 'meta', None)
        uid = gws.get(meta, 'iso.uid')
        if not uid:
            continue
        m = _configure_metadata(obj)
        if m:
            rs[gws.as_uid(uid)] = m
    return rs


def _configure_metadata(obj: t.ObjectInterface):
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
        m.resolution = int(min(gws.tools.misc.res2scale(r) for r in la.resolutions))

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


##

def _layer_node_roots(rd: RequestData) -> t.List[LayerCapsNode]:
    if not rd.project:
        return []
    return gws.compact(_layer_node_subtree(rd, la.uid) for la in rd.project.map.layers)


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
        'title': layer.title,
        'tag_name': layer.ows_name,
        'has_search': layer.has_search or any(n.has_search for n in sub_nodes),
        'meta': layer.meta,
        'feature_schema': None,
        'extent': layer.extent,
        'lonlat_extent': lonlat_extent(layer.extent, crs),
        'min_scale': min(res),
        'max_scale': max(res),
        'proj': gws.gis.proj.as_proj(crs),
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
