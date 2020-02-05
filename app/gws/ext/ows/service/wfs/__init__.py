import gws
import gws.common.model
import gws.common.ows.service as ows
import gws.common.search.runner
import gws.gis.extent
import gws.gis.gml
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.tools.shell
import gws.tools.xml3
import gws.web.error

import gws.types as t

_xml_types = {
    t.AttributeType.bool: 'xsd:boolean',
    t.AttributeType.bytes: None,
    t.AttributeType.date: 'xsd:date',
    t.AttributeType.datetime: 'datetime',
    t.AttributeType.float: 'xsd:decimal',
    t.AttributeType.geometry: None,
    t.AttributeType.int: 'xsd:integer',
    t.AttributeType.list: None,
    t.AttributeType.str: 'xsd:string',
    t.AttributeType.text: 'xsd:string',
    t.AttributeType.time: 'xsd:time',
    t.GeometryType.curve: 'gml:CurvePropertyType',
    t.GeometryType.geomcollection: 'gml:MultiGeometryPropertyType',
    t.GeometryType.geometry: 'gml:MultiGeometryPropertyType',
    t.GeometryType.linestring: 'gml:CurvePropertyType',
    t.GeometryType.multicurve: 'gml:MultiCurvePropertyType',
    t.GeometryType.multilinestring: 'gml:MultiCurvePropertyType',
    t.GeometryType.multipoint: 'gml:MultiPointPropertyType',
    t.GeometryType.multipolygon: 'gml:MultiGeometryPropertyType',
    t.GeometryType.multisurface: 'gml:MultiGeometryPropertyType',
    t.GeometryType.point: 'gml:PointPropertyType',
    t.GeometryType.polygon: 'gml:SurfacePropertyType',
    t.GeometryType.polyhedralsurface: 'gml:SurfacePropertyType',
    t.GeometryType.surface: 'gml:SurfacePropertyType',
}


class Config(ows.Config):
    """WFS Service configuration"""

    pass


VERSION = '2.0'
MAX_LIMIT = 100


class Object(ows.Base):
    def __init__(self):
        super().__init__()

        self.type = 'wfs'
        self.version = VERSION

    @property
    def service_link(self):
        return t.MetaLink({
            'url': self.url,
            'scheme': 'OGC:WFS',
            'function': 'download'
        })

    def configure(self):
        super().configure()

        for tpl in 'getCapabilities', 'describeFeatureType', 'getFeature', 'feature':
            self.templates[tpl] = self.configure_template(tpl, 'wfs/templates')

    def handle_getcapabilities(self, rd: ows.Request):
        nodes = self.layer_node_list(rd)
        if self.use_inspire_data:
            nodes = self.inspire_nodes(nodes)
        return self.xml_response(self.render_template(rd, 'getCapabilities', {
            'layer_node_list': nodes,
        }))

    def handle_describefeaturetype(self, rd: ows.Request):
        nodes = self._nodes_from_request(rd)

        if self.use_inspire_data:
            return self._describe_inspire_features(nodes)

        for node in nodes:
            dm = node.layer.data_model
            if not dm:
                continue

            node.feature_schema = []
            for rule in dm.rules:
                xtype = _xml_types.get(rule.type)
                if xtype:
                    node.feature_schema.append({
                        'name': rule.name,
                        'type': xtype
                    })

            if dm.geometry_type:
                node.feature_schema.append({
                    'name': 'geometry',
                    'type': _xml_types.get(dm.geometry_type)
                })

        return self.xml_response(self.render_template(rd, 'describeFeatureType', {
            'layer_node_list': nodes,
        }))

    def _describe_inspire_features(self, nodes):
        # @TODO inspire schemas
        pass

    def handle_getfeature(self, rd: ows.Request):
        nodes = self._nodes_from_request(rd)
        try:
            limit = int(rd.req.param('count') or rd.req.param('maxFeatures') or MAX_LIMIT)
        except:
            raise gws.web.error.BadRequest()

        bbox = None
        if rd.req.has_param('bbox'):
            bbox = gws.gis.extent.from_string(rd.req.param('bbox'))
            if not bbox:
                raise gws.web.error.BadRequest()

        shape = gws.gis.shape.from_extent(
            extent=bbox or rd.project.map.extent,
            crs=rd.req.param('srsName') or rd.project.map.crs
        )

        args = t.SearchArgs(
            project=rd.project,
            shapes=[shape],
            layers=[n.layer for n in nodes],
            limit=min(limit, MAX_LIMIT),
            tolerance=(10, 'px'),
            resolution=1,
        )

        features = gws.common.search.runner.run(rd.req, args)
        nodes = self.feature_node_list(rd, features)
        return self.render_feature_nodes(rd, nodes, 'getFeature')

    def _nodes_from_request(self, rd):
        nodes = self.layer_nodes_from_request_params(rd, ['typeName', 'typeNames'])

        if self.use_inspire_data:
            nodes = self.inspire_nodes(nodes)

        if not nodes:
            raise gws.web.error.NotFound()

        return nodes
