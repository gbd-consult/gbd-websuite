import gws
import gws.common.search.runner
import gws.common.ows.service
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.ows.gml
import gws.tools.shell
import gws.tools.xml3
import gws.web.error
import gws.tools.net
import gws.tools.misc

import gws.types as t

import gws.common.ows.service as ows
import gws.common.ows.service.inspire as inspire


class TemplatesConfig(t.Config):
    """WFS service templates"""

    getCapabilities: t.Optional[t.TemplateConfig]  #: xml template for the WFS capabilities document
    getFeature: t.Optional[t.TemplateConfig]  #: xml template for the WFS GetFeature document


class ThemeConfig(t.Config):
    name: str
    templates: t.Optional[TemplatesConfig]


class Config(ows.Config):
    """WFS Service configuration"""

    templates: t.Optional[TemplatesConfig]  #: service templates


VERSION = '2.0'
MAX_LIMIT = 100


class Object(ows.Object):
    def __init__(self):
        super().__init__()

        self.service_class = 'inspirewfs'
        self.service_type = 'wfs'
        self.version = VERSION

    def configure(self):
        super().configure()

        for tpl in 'getCapabilities', 'getFeature':
            self.templates[tpl] = self.configure_template(tpl, 'inspirewfs/templates')
        configure_inspire_templates(self)

    def handle_getcapabilities(self, rd: ows.RequestData):
        nodes = inspire_nodes(ows.layer_node_list(rd))
        if not nodes:
            raise gws.web.error.NotFound()

        return ows.xml_response(self.render_template(rd, 'getCapabilities', {
            'layer_node_list': nodes,
        }))

    def handle_describefeaturetype(self, rd: ows.RequestData):
        nodes = inspire_nodes(ows.layer_nodes_from_request_params(rd, 'typeName', 'typeNames'))
        if not nodes:
            raise gws.web.error.NotFound()

        namespaces = set(n.tag_name.split(':')[0] for n in nodes)

        for ns in namespaces:
            # @TODO combine schemas?
            _, schema_location = inspire.NAMESPACES[ns]
            res = gws.tools.net.http_request(schema_location, max_age=3600 * 24 * 30)
            return t.HttpResponse({
                'mimeType': 'text/xml',
                'content': res.text,
                'status': 200,
            })

    def handle_getfeature(self, rd: ows.RequestData):
        nodes = inspire_nodes(ows.layer_nodes_from_request_params(rd, 'typeName', 'typeNames'))
        if not nodes:
            raise gws.web.error.NotFound()

        try:
            limit = int(rd.req.kparam('count') or rd.req.kparam('maxFeatures') or MAX_LIMIT)
            bbox = rd.project.map.extent
            if rd.req.kparam('bbox'):
                bbox = [float(n) for n in rd.req.kparam('bbox').split(',')[:4]]
        except:
            raise gws.web.error.BadRequest()

        args = t.SearchArgs({
            'shapes': [gws.gis.shape.from_bbox(bbox, rd.project.map.crs)],
            'crs': rd.project.map.crs,
            'project': None,
            'keyword': None,
            'layers': [n.layer for n in nodes],
            'limit': min(limit, MAX_LIMIT),
            'tolerance': 10,
        })

        features = gws.common.search.runner.run(rd.req, args)
        return render_inspire_features(rd, features, 'getFeature')


def inspire_nodes(nodes):
    return [n for n in nodes if n.tag_name in inspire.TAGS]


def configure_inspire_templates(service):
    for tag in inspire.TAGS:
        service.templates[tag] = service.configure_template(tag.replace(':', '_'), 'inspirewfs/templates/themes')


def render_inspire_features(rd: ows.RequestData, features, container_template):
    nodes = inspire_nodes(ows.feature_node_list(rd, features))
    for n in nodes:
        n.template_name = n.tag_name
    return ows.render_feature_nodes(rd, nodes, container_template)
