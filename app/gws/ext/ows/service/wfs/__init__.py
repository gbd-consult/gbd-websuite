import os

import gws
import gws.common.search.runner
import gws.common.ows.service
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.ows.gml
import gws.tools.misc as misc
import gws.tools.shell
import gws.tools.xml3
import gws.web.error

import gws.types as t

# NB templates use "xsd" as a schema namespace

_ATTR_TYPE_TO_XML = {
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


class TemplatesConfig(t.Config):
    """WFS service templates"""

    getCapabilities: t.Optional[t.TemplateConfig]  #: xml template for the WFS capabilities document
    describeFeatureType: t.Optional[t.TemplateConfig]  #: xml template for the WFS DescribeFeatureType document
    getFeature: t.Optional[t.TemplateConfig]  #: xml template for the WFS GetFeature document


class Config(gws.common.ows.service.Config):
    """WFS Service configuration"""

    templates: t.Optional[TemplatesConfig]  #: service templates


VERSION = '2.0'
MAX_LIMIT = 100


class Object(gws.common.ows.service.Object):
    def configure(self):
        super().configure()

        for tpl in 'getCapabilities', 'describeFeatureType', 'getFeature':
            self.templates[tpl] = self.configure_template(
                tpl,
                os.path.dirname(__file__))

    def error_response(self, status):
        return self.xml_error_response(VERSION, status, f'Error {status}')

    def handle(self, req):
        project = req.require_project(req.param('projectUid'))

        r = req.kparam('request', '').lower()

        if r == 'getcapabilities':
            return self.handle_getcapabilities(req, project)
        if r == 'describefeaturetype':
            return self.handle_describefeaturetype(req, project)
        if r == 'getfeature':
            return self.handle_getfeature(req, project)

        raise gws.web.error.NotFound()

    def handle_getcapabilities(self, req, project):
        return self.render_template(req, project, 'getCapabilities', {
            'layer_node_list': self.layer_node_list(req, project),
        })

    def handle_describefeaturetype(self, req, project):
        nodes = self._nodes_from_type_names(req, project)

        for node in nodes:
            node.schema = []
            for a in node.layer.data_model:
                xtype = _ATTR_TYPE_TO_XML.get(a.type)
                if xtype:
                    node.schema.append({
                        'name': 'geometry' if a.type.startswith('geo') else gws.as_uid(a.title),
                        'type': xtype
                    })

        return self.render_template(req, project, 'describeFeatureType', {
            'layer_node_list': nodes,
        })

    def handle_getfeature(self, req, project):
        try:
            limit = int(req.kparam('count') or req.kparam('maxFeatures') or MAX_LIMIT)
            bbox = None
            if req.kparam('bbox'):
                bbox = [float(n) for n in req.kparam('bbox').split(',')[:4]]
        except:
            raise gws.web.error.BadRequest()

        nodes = self._nodes_from_type_names(req, project)

        if not bbox:
            bbox = project.map.extent

        args = t.SearchArgs({
            'shapes': [gws.gis.shape.from_bbox(bbox, project.map.crs)],
            'crs': project.map.crs,
            'project': None,
            'keyword': None,
            'layers': [n.layer for n in nodes],
            'limit': min(limit, MAX_LIMIT),
            'tolerance': 10,
        })

        features = gws.common.search.runner.run(req, args)
        return self.render_template(req, project, 'getFeature', {
            'feature_nodes': self.feature_node_list(req, project, features)
        })

    def _nodes_from_type_names(self, req, project):
        ows_names = req.kparam('typeNames') or req.kparam('typeName')
        nodes = self.layer_node_list(
            req, project, ows_names.split(',') if ows_names else None)
        if not nodes:
            raise gws.web.error.NotFound()
        return nodes
