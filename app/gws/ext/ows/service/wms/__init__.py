import gws
import gws.common.search.runner
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.ows.gml
import gws.tools.misc as misc
import gws.tools.shell
import gws.tools.xml3
import gws.web.error

import gws.types as t

import gws.common.ows.service as ows


class Config(gws.common.ows.service.Config):
    """WMS Service configuration"""
    pass


VERSION = '1.3.0'
MAX_LIMIT = 100


class Object(ows.Object):
    def __init__(self):
        super().__init__()

        self.type = 'wms'
        self.version = VERSION

    @property
    def service_link(self):
        return t.MetaLink({
            'url': self.service_url,
            'scheme': 'OGC:WMS',
            'function': 'search'
        })

    def configure(self):
        super().configure()

        for tpl in 'getCapabilities', 'getFeatureInfo', 'feature':
            self.templates[tpl] = self.configure_template(tpl, 'wms/templates')

    def handle_getcapabilities(self, rd: ows.RequestData):
        root = ows.layer_node_root(rd)
        if not root:
            raise gws.web.error.NotFound()

        return ows.xml_response(self.render_template(rd, 'getCapabilities', {
            'layer_node_root': root,
        }))

    def handle_getmap(self, rd: ows.RequestData):
        try:
            bbox = [float(n) for n in rd.req.kparam('bbox').split(',')]
            px_width = int(rd.req.kparam('width'))
            px_height = int(rd.req.kparam('height'))
        except:
            raise gws.web.error.BadRequest()

        nodes = ows.layer_nodes_from_request_params(rd, 'layers', 'layer')
        if not nodes:
            raise gws.web.error.NotFound()

        render_input = t.MapRenderInput({
            'out_path': gws.TMP_DIR + '/wms_' + gws.random_string(64) + '.png',
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

    def handle_getlegendgraphic(self, rd: ows.RequestData):
        # https://docs.geoserver.org/stable/en/user/services/wms/get_legend_graphic/index.html
        # @TODO currently only support 'layer'

        nodes = ows.layer_nodes_from_request_params(rd, 'layers', 'layer')
        if not nodes:
            raise gws.web.error.NotFound()

        img = nodes[0].layer.render_legend()

        return t.HttpResponse({
            'mimeType': 'image/png',
            'content': img or gws.tools.misc.Pixels.png8
        })

    def handle_getfeatureinfo(self, rd: ows.RequestData):
        features = find_features(rd)
        nodes = ows.feature_node_list(rd, features)
        return self.render_feature_nodes(rd, nodes, 'getFeatureInfo')


def find_features(rd: ows.RequestData):
    try:
        bbox = [float(n) for n in rd.req.kparam('bbox').split(',')]
        px_width = int(rd.req.kparam('width'))
        px_height = int(rd.req.kparam('height'))
        limit = int(rd.req.kparam('feature_count', '1'))
        i = int(rd.req.kparam('i'))
        j = int(rd.req.kparam('j'))
    except:
        raise gws.web.error.BadRequest()

    nodes = ows.layer_nodes_from_request_params(rd, 'query_layers')
    if not nodes:
        raise gws.web.error.NotFound()

    xres = (bbox[2] - bbox[0]) / px_width
    yres = (bbox[3] - bbox[1]) / px_height
    x = bbox[0] + (i * xres)
    y = bbox[3] - (j * yres)

    point = gws.gis.shape.from_props(t.ShapeProps({
        'crs': rd.project.map.crs,
        'geometry': {
            'type': 'Point',
            'coordinates': [x, y]
        }}
    ))

    pixel_tolerance = 10

    args = t.SearchArgs({
        'bbox': bbox,
        'crs': rd.project.map.crs,
        'project': None,
        'keyword': None,
        'layers': [n.layer for n in nodes],
        'limit': min(limit, MAX_LIMIT),
        'resolution': xres,
        'shapes': [point],
        'tolerance': pixel_tolerance * xres,
    })

    return gws.common.search.runner.run(rd.req, args)
