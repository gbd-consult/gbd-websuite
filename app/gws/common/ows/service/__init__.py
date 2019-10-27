import gws
import gws.web.error
import gws.tools.misc
import gws.tools.xml3
import gws.gis.proj
import gws.ows.gml
import gws.types as t


class Config(t.WithTypeAndAccess):
    enabled: bool = True
    xmlNamespace: str = 'gws'  #: feature namespace name
    xmlNamespaceUri: str = 'https://gws.gbd-consult.de'  #: feature namespace uri


class Object(gws.Object):
    """Generic OWS Service."""

    def __init__(self):
        super().__init__()
        self.templates = {}

    def can_handle(self, req) -> bool:
        return req.kparam('service', '').lower() == self.var('type')

    def handle(self, req) -> t.HttpResponse:
        raise gws.web.error.NotFound

    def error_response(self, status) -> t.HttpResponse:
        raise gws.web.error.NotFound

    def configure_template(self, name, base_path):
        p = self.var('templates.' + name)
        if p:
            return self.create_object('gws.ext.template', p)

        return self.create_shared_object('gws.ext.template', base_path + name, {
            'type': 'xml',
            'path': base_path + f'/templates/{name}.cx'
        })

    def xml_response(self, content, status=200):
        return t.HttpResponse({
            'mimeType': 'text/xml',
            'content': gws.tools.xml3.as_string(content, compress=True),
            'status': status,
        })

    def xml_error_response(self, version, status, description):
        description = gws.tools.xml3.encode(description)
        content = f'<?xml version="1.0" encoding="UTF-8"?>' \
                  f'<ServiceExceptionReport version="{version}">' \
                  f'<ServiceException code="{status}">{description}</ServiceException>' \
                  f'</ServiceExceptionReport>'
        return self.xml_response(content, status)

    def layer_node_tree(self, req, project):
        return gws.compact(self.layer_node_subtree(req, la.uid) for la in project.map.layers)

    def layer_node_subtree(self, req, layer_uid):
        layer = req.acquire('gws.ext.layer', layer_uid)

        if not self.is_layer_enabled(layer):
            return

        if not layer.layers:
            return self.layer_node(layer)

        sub = gws.compact(self.layer_node_subtree(req, la.uid) for la in layer.layers)
        if sub:
            return self.layer_node(layer, sub)

    def layer_node(self, layer, sub_nodes=None):
        res = [gws.tools.misc.res2scale(r) for r in layer.resolutions]
        crs = layer.map.crs
        sub_nodes = sub_nodes or []

        return t.Data({
            'layer': layer,
            'extent': layer.extent,
            'lonlat_extent': ['%.3f1' % c for c in gws.gis.proj.transform_bbox(layer.extent, crs, 'EPSG:4326')],
            'proj': gws.gis.proj.as_proj(crs),
            'has_search': layer.has_search or any(s['has_search'] for s in sub_nodes),
            'min_scale': min(res),
            'max_scale': max(res),
            'sub_nodes': sub_nodes,
        })

    def layer_node_list(self, req, project, ows_names=None):
        # strip namespaces from ows_names
        if ows_names:
            ows_names = [n.split(':')[-1] for n in ows_names]

        all_nodes = []
        for n in self.layer_node_tree(req, project):
            self.layer_node_sublist(req, n, all_nodes)
        if ows_names:
            all_nodes = [n for n in all_nodes if n.layer.ows_name in ows_names]
        return all_nodes

    def layer_node_sublist(self, req, node, all_nodes):
        if not node.sub_nodes:
            all_nodes.append(node)
            return
        for n in node.sub_nodes:
            self.layer_node_sublist(req, n, all_nodes)

    def feature_node_list(self, req, project, features):
        return [self.feature_node(project, f) for f in features]

    def feature_node(self, project, feature):
        gs = None
        if feature.shape:
            gs = gws.tools.xml3.as_string(gws.ows.gml.shape_to_tag(feature.shape, precision=project.map.coordinate_precision))

        return t.Data({
            'feature': feature,
            'gml_shape': gs,
            'attributes': {gws.as_uid(k): v for k, v in feature.attributes.items()}
        })

    def is_layer_enabled(self, layer):
        return layer and layer.has_ows(self.var('type'))

    def render_template(self, req, project, template, context):
        context = gws.extend(context, {
            'project': project,
            'namespace': self.var('xmlNamespace'),
            'namespace_uri': self.var('xmlNamespaceUri'),
            'service_endpoint': req.reversed_url(f'cmd=owsHttpGet&projectUid={project.uid}'),
        })
        out = self.templates[template].render(context)
        return self.xml_response(out.content)
