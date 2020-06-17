import gws
import gws.common.metadata
import gws.common.metadata.inspire
import gws.common.model
import gws.common.search.runner
import gws.gis.extent
import gws.gis.gml
import gws.gis.proj
import gws.tools.units as units
import gws.tools.xml2
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    featureNamespace: str = ''  #: feature namespace name and uri
    meta: t.Optional[gws.common.metadata.Config]  #: service metadata
    templates: t.Optional[t.List[t.ext.template.Config]]  #: service XML templates
    withInspireMeta: bool = False  #: use INSPIRE Metadata
    withInspireData: bool = False  #: use INSPIRE data model


class Request(t.Data):
    req: t.IRequest
    project: t.IProject
    service: t.IOwsService
    xml: gws.tools.xml2.Element = None
    xml_is_soap: bool = False


class LayerCapsNode(t.Data):
    layer: t.ILayer = None

    feature_schema: t.List[t.Attribute] = []
    has_legend: bool = False
    has_search: bool = False
    meta: t.MetaData = None
    tag_name: str = None
    title: str = None

    extent: t.Extent = None
    geographic_extent: t.Extent = None
    max_scale: int = None
    min_scale: int = None
    proj: gws.gis.proj.Proj = None

    sub_nodes: t.Optional[t.List['LayerCapsNode']] = []


class FeatureNode(t.Data):
    feature: t.IFeature
    shape_tag: ''
    tag_name: ''
    attributes: t.List[t.Attribute]


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

    def error_response(self, status) -> t.HttpResponse:
        pass


class Base(Object):
    """Baseclass for OWS services."""

    @property
    def url(self):
        u = gws.SERVER_ENDPOINT + '/cmd/owsHttpService/uid/' + self.uid
        if self.project:
            u += f'/projectUid/{self.project.uid}'
        return u

    @property
    def service_link(self):
        return None

    # Configuration

    def configure(self):
        super().configure()

        self.local_namespaces = {}
        self.project: t.Optional[t.IProject] = t.cast(t.IProject, self.get_closest('gws.common.project'))

        self.templates = {}
        self.use_inspire_data = False
        self.use_inspire_meta = False

        self.feature_namespace = ''

        s = self.var('featureNamespace')
        if s:
            self.feature_namespace = self.var('featureNamespace')
            self.local_namespaces[self.feature_namespace] = self.var('featureNamespaceUri')

        self.use_inspire_meta = self.var('withInspireMeta')
        self.use_inspire_data = self.var('withInspireData')

        self.meta = self.configure_metadata()

        if self.use_inspire_data:
            self.configure_inspire_templates()

    def post_configure(self):
        super().post_configure()
        self.configure_metadata()

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
            serviceUrl=self.url,
        )

        if self.service_link:
            meta.links.append(self.service_link)

        return meta

    def configure_template(self, name, path, type='xml'):
        for tpl in self.var('templates', default=[]):
            if tpl.title == name:
                return self.root.create_object('gws.ext.template', tpl)

        if not path.startswith('/'):
            path = gws.APP_DIR + '/gws/ext/ows/service/' + path.lstrip('/')
        if path.endswith('/'):
            path += name + '.cx'

        return self.root.create_shared_object('gws.ext.template', path, {
            'type': type,
            'path': path
        })

    def configure_inspire_templates(self):
        for tag in gws.common.metadata.inspire.TAGS:
            self.templates[tag] = self.configure_template(tag.replace(':', '_'), 'wfs/templates/inspire/')

    # Request handling

    def handle(self, req) -> t.HttpResponse:
        # services can be configured globally (in which case, self.project=None)
        # and applied to multiple projects with the projectUid param

        project = None

        p = req.param('projectUid')
        if p:
            project = req.require_project(p)
            if self.project and project != self.project:
                gws.log.debug(f'service={self.uid!r}: wrong project={p!r}')
                raise gws.web.error.NotFound()
        elif self.project:
            # for in-project services, ensure the user can access the project
            req.require_project(self.project.uid)

        rd = Request({
            'req': req,
            'project': project or self.project,
        })

        return self.dispatch(rd, req.param('request', ''))

    def dispatch(self, rd: Request, request_param):
        h = getattr(self, 'handle_' + request_param.lower(), None)
        if not h:
            gws.log.debug(f'service={self.uid!r}: request={request_param!r} not found')
            raise gws.web.error.NotFound()
        return h(rd)

    # Rendering and responses

    def error_response(self, status):
        return self.xml_error_response(self.version, status, f'Error {status}')

    def render_template(self, rd: Request, template_name: str, context=None, format=None):
        context = gws.merge({
            'project': rd.project,
            'meta': self.meta,
            'use_inspire_meta': self.use_inspire_meta,
            'url_for': rd.req.url_for,
            'feature_namespace': self.feature_namespace,
            'feature_namespace_uri': self.local_namespaces.get(self.feature_namespace, ''),
            'namespaces': self.local_namespaces,
            'service': self
        }, context)

        return self.templates[template_name].render(context, format=format).content

    def render_feature_nodes(self, rd: Request, nodes: t.List[FeatureNode], container_template_name: str) -> t.HttpResponse:
        tags = []

        for node in nodes:
            template_name = node.tag_name if node.tag_name in self.templates else 'feature'
            tags.append(self.render_template(rd, template_name, {'node': node}, format='tag'))

        context = {
            'feature_tags': tags,
        }

        return self.xml_response(self.render_template(rd, container_template_name, context))

    def xml_error_response(self, version, status, description) -> t.HttpResponse:
        description = gws.tools.xml2.encode(description)
        content = (f'<?xml version="1.0" encoding="UTF-8"?>'
                   + f'<ServiceExceptionReport version="{version}">'
                   + f'<ServiceException code="{status}">{description}</ServiceException>'
                   + f'</ServiceExceptionReport>')
        return self.xml_response(content, status)

    def xml_response(self, content, status=200) -> t.HttpResponse:
        return t.HttpResponse({
            'mime': 'text/xml',
            'content': gws.tools.xml2.as_string(content),
            'status': status,
        })

    # LayerCaps nodes

    def layer_tree_root(self, rd: Request) -> t.Optional[LayerCapsNode]:
        """Return a single root node for a layer tree."""

        if not rd.project:
            return

        roots = gws.compact(self._layer_node_subtree(rd, la.uid) for la in rd.project.map.layers)

        if not roots:
            return

        return LayerCapsNode(
            extent=rd.project.map.extent,
            has_legend=any(n.has_legend for n in roots),
            has_search=any(n.has_search for n in roots),
            meta=rd.project.meta,
            sub_nodes=roots,
            tag_name=rd.project.uid,
            title=rd.project.title,
        )

    def layer_node_list(self, rd: Request) -> t.List[LayerCapsNode]:
        """Return a list of terminal layer nodes (for WFS)."""

        all_nodes = []
        root = self.layer_tree_root(rd)
        if not root:
            return []
        self._layer_node_sublist(rd, root, all_nodes)
        return all_nodes

    def layer_nodes_from_request_params(self, rd: Request, param_names, fallback_to_all=True):
        """Return a list of terminal layer nodes matching the layer list."""

        names = None

        for p in param_names:
            if rd.req.has_param(p):
                names = gws.as_list(rd.req.param(p))
                break

        if names is None and fallback_to_all:
            return self.layer_node_list(rd)

        if not names:
            return []

        tag_names = set()

        for s in names:
            if ':' not in s:
                tag_names.add(s)
                continue

            p = s.split(':')
            if len(p) != 2:
                continue

            if p[0] == self.feature_namespace:
                tag_names.add(s)
                tag_names.add(p[1])
            else:
                tag_names.add(s)

        all_nodes = []
        self._layer_node_sublist_selected(rd, self.layer_tree_root(rd), all_nodes, tag_names)
        return all_nodes

    def _layer_node_subtree(self, rd: Request, layer_uid):
        layer = t.cast(t.ILayer, rd.req.acquire('gws.ext.layer', layer_uid))
        if not self.is_layer_enabled(layer):
            return
        if not layer.layers:
            return self._layer_node_from(layer)
        sub = gws.compact(self._layer_node_subtree(rd, la.uid) for la in layer.layers)
        if sub:
            return self._layer_node_from(layer, sub)

    def _layer_node_sublist(self, rd: Request, node: LayerCapsNode, all_nodes):
        if not node.sub_nodes:
            all_nodes.append(node)
            return
        for n in node.sub_nodes:
            self._layer_node_sublist(rd, n, all_nodes)

    def _layer_node_sublist_selected(self, rd: Request, node: LayerCapsNode, all_nodes, tag_names):
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

        return LayerCapsNode(
            extent=layer.extent,
            # in a service, only provide legends for leaf layers
            has_legend=layer.has_legend and not sub_nodes,
            has_search=layer.has_search or any(n.has_search for n in sub_nodes),
            layer=layer,
            meta=layer.meta,
            sub_nodes=sub_nodes,
            tag_name=layer.ows_name,
            title=layer.title,
        )

    def inspire_nodes(self, nodes):
        return [n for n in nodes if n.tag_name in gws.common.metadata.inspire.TAGS]

    def feature_node_list(self, rd: Request, features: t.List[t.IFeature]) -> t.List[FeatureNode]:
        def node(f: t.IFeature):
            gs = None
            if f.shape:
                gs = gws.gis.gml.shape_to_tag(f.shape, precision=rd.project.map.coordinate_precision)

            f.apply_data_model()

            return FeatureNode(
                feature=f,
                shape_tag=gs,
                tag_name=f.layer.ows_name if f.layer else 'feature',
                attributes=f.attributes,
            )

        return [node(f) for f in features]

    # Utils

    def is_layer_enabled(self, layer):
        return layer and layer.ows_enabled(self)
