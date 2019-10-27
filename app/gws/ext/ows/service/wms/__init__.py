import re
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


class TemplatesConfig(t.Config):
    getCapabilities: t.Optional[t.TemplateConfig]  #: xml template for the WMS capabilities document
    getFeatureInfo: t.Optional[t.TemplateConfig]  #: xml template for the WMS GetFeatureInfo response


class Config(gws.common.ows.service.Config):
    """WMS Service configuration"""

    templates: t.Optional[TemplatesConfig]  #: service templates


VERSION = '1.3.0'
MAX_LIMIT = 100


class Object(gws.common.ows.service.Object):
    def configure(self):
        super().configure()
        for tpl in 'getCapabilities', 'getFeatureInfo':
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
        if r == 'getmap':
            return self.handle_getmap(req, project)
        if r == 'getfeatureinfo':
            return self.handle_getfeatureinfo(req, project)
        if r == 'getlegendgraphic':
            return self.handle_getlegendgraphic(req, project)

        raise gws.web.error.NotFound()

    def handle_getcapabilities(self, req, project):
        return self.render_template(req, project, 'getCapabilities', {
            'layer_node_tree': self.layer_node_tree(req, project),
        })

    def handle_getmap(self, req, project):
        try:
            ows_names = req.kparam('layers').split(',')
            bbox = [float(n) for n in req.kparam('bbox').split(',')]
            px_width = int(req.kparam('width'))
            px_height = int(req.kparam('height'))
        except:
            raise gws.web.error.BadRequest()

        nodes = self.layer_node_list(req, project, ows_names)
        if not nodes:
            raise gws.web.error.NotFound()

        render_input = t.MapRenderInput({
            'out_path': '/tmp/wms_' + gws.random_string(64) + '.png',
            'bbox': bbox,
            'rotation': 0,
            'scale': gws.tools.misc.res2scale((bbox[2] - bbox[0]) / px_width),
            'dpi': 0,
            'map_size_px': [px_width, px_height],
            'background_color': 0,
            'items': [],
        })

        for node in nodes:
            item = t.MapRenderInputItem({
                'layer': node.layer,
                'sub_layers': []
            })
            render_input.items.append(item)

        renderer = gws.gis.render.Renderer()
        for _ in renderer.run(render_input):
            pass

        with open(renderer.output.items[0].image_path, 'rb') as fp:
            img = fp.read()

        gws.tools.shell.unlink(renderer.output.items[0].image_path)

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img
        })

    def handle_getlegendgraphic(self, req, project):
        # https://docs.geoserver.org/stable/en/user/services/wms/get_legend_graphic/index.html
        # @TODO currently only support 'layer'

        ows_name = req.kparam('layer')
        nodes = self.layer_node_list(req, project, [ows_name])
        if not nodes:
            raise gws.web.error.NotFound()

        img = nodes[0].layer.render_legend()

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img or gws.tools.misc.Pixels.png8
        })

    def handle_getfeatureinfo(self, req, project):
        try:
            ows_names = req.kparam('query_layers').split(',')
            bbox = [float(n) for n in req.kparam('bbox').split(',')]
            px_width = int(req.kparam('width'))
            px_height = int(req.kparam('height'))
            limit = int(req.kparam('feature_count', '1'))
            i = int(req.kparam('i'))
            j = int(req.kparam('j'))
        except:
            raise gws.web.error.BadRequest()

        nodes = self.layer_node_list(req, project, ows_names)
        if not nodes:
            raise gws.web.error.NotFound()

        xres = (bbox[2] - bbox[0]) / px_width
        yres = (bbox[3] - bbox[1]) / px_height
        x = bbox[0] + (i * xres)
        y = bbox[3] - (j * yres)

        gws.p('WMS_POINT', bbox, i, j, xres, yres, x, y)

        point = gws.gis.shape.from_props(t.ShapeProps({
            'crs': project.map.crs,
            'geometry': {
                'type': 'Point',
                'coordinates': [x, y]
            }}
        ))

        pixel_tolerance = 10

        args = t.SearchArgs({
            'bbox': bbox,
            'crs': project.map.crs,
            'project': None,
            'keyword': None,
            'layers': [n.layer for n in nodes],
            'limit': min(limit, MAX_LIMIT),
            'resolution': xres,
            'shapes': [point],
            'tolerance': pixel_tolerance * xres,
        })

        features = gws.common.search.runner.run(req, args)
        return self.render_template(req, project, 'getFeatureInfo', {
            'feature_nodes': self.feature_node_list(req, project, features)
        })

    def gml_bounded_by(self, features):
        return ''

    def gml_feature(self, f: t.FeatureInterface, project):
        def _as_ident(s):
            return re.sub(r'\W+', '_', s)

        tag = gws.tools.xml3.tag

        props = [tag(_as_ident(k), v) for k, v in f.attributes.items()]
        if f.shape:
            props.append(tag(
                'geometry',
                gws.ows.gml.shape_to_tag(f.shape, precision=project.map.coordinate_precision)))
        return tag('wfs:member', tag(f.category, {'gml:id': f.uid}, *props))
