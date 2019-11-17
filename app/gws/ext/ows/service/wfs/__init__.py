import gws
import gws.common.datamodel
import gws.common.ows.service
import gws.common.search.runner
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.ows.gml
import gws.tools.shell
import gws.tools.xml3
import gws.web.error

import gws.types as t

import gws.common.ows.service as ows


class TemplatesConfig(t.Config):
    """WFS service templates"""

    getCapabilities: t.Optional[t.TemplateConfig]  #: xml template for the WFS capabilities document
    describeFeatureType: t.Optional[t.TemplateConfig]  #: xml template for the WFS DescribeFeatureType document
    getFeature: t.Optional[t.TemplateConfig]  #: xml template for the WFS GetFeature document
    feature: t.Optional[t.TemplateConfig]  #: xml template for a WFS feature


class Config(gws.common.ows.service.Config):
    """WFS Service configuration"""

    templates: t.Optional[TemplatesConfig]  #: service templates


VERSION = '2.0'
MAX_LIMIT = 100


class Object(ows.Object):
    def __init__(self):
        super().__init__()

        self.service_class = 'wfs'
        self.service_type = 'wfs'
        self.version = VERSION

    def configure(self):
        super().configure()

        for tpl in 'getCapabilities', 'describeFeatureType', 'getFeature', 'feature':
            self.templates[tpl] = self.configure_template(tpl, 'wfs/templates')

    def handle_getcapabilities(self, rd: ows.RequestData):
        return ows.xml_response(self.render_template(rd, 'getCapabilities', {
            'layer_node_list': ows.layer_node_list(rd),
        }))

    def handle_describefeaturetype(self, rd: ows.RequestData):
        nodes = ows.layer_nodes_from_request_params(rd, 'typeName', 'typeNames')
        if not nodes:
            raise gws.web.error.NotFound()

        for node in nodes:
            dm = node.layer.data_model
            node.feature_schema = []
            for a in dm:
                xtype = ows.ATTR_TYPE_TO_XML.get(a.type or 'str')
                if xtype:
                    node.feature_schema.append({
                        'name': a.name,
                        'type': xtype
                    })

            gtype = node.layer.geometry_type
            if gtype:
                node.feature_schema.append({
                    'name': 'geometry',
                    'type': ows.ATTR_TYPE_TO_XML.get(gtype)
                })

        return ows.xml_response(self.render_template(rd, 'describeFeatureType', {
            'layer_node_list': nodes,
        }))

    def handle_getfeature(self, rd: ows.RequestData):
        nodes = ows.layer_nodes_from_request_params(rd, 'typeName', 'typeNames')
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
        nodes = ows.feature_node_list(rd, features)
        return ows.render_feature_nodes(rd, nodes, 'getFeature')
