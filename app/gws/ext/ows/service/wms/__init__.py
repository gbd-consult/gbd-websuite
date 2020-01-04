import io

import gws
import gws.common.search.runner
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.gis.extent
import gws.gis.gml
import gws.tools.misc as misc
import gws.tools.units as units
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


class Object(ows.Base):
    def __init__(self):
        super().__init__()

        self.type = 'wms'
        self.version = VERSION

    @property
    def service_link(self):
        return t.MetaLink(
            url=self.url,
            scheme='OGC:WMS',
            function='search'
        )

    def configure(self):
        super().configure()

        for tpl in 'getCapabilities', 'getFeatureInfo', 'feature':
            self.templates[tpl] = self.configure_template(tpl, 'wms/templates')

    def handle_getcapabilities(self, rd: ows.OwsRequest):
        root = self.layer_node_root(rd)
        if not root:
            raise gws.web.error.NotFound()
        return self.xml_response(self.render_template(rd, 'getCapabilities', {
            'layer_node_root': root,
        }))

    def handle_getmap(self, rd: ows.OwsRequest):
        try:
            bbox = gws.gis.extent.from_string(rd.req.kparam('bbox'))
            px_width = int(rd.req.kparam('width'))
            px_height = int(rd.req.kparam('height'))
        except:
            raise gws.web.error.BadRequest()

        if not bbox or not px_width or not px_height:
            raise gws.web.error.BadRequest()

        nodes = self.layer_nodes_from_request_params(rd, 'layers', 'layer')
        if not nodes:
            raise gws.web.error.NotFound()

        render_input = t.RenderInput(
            background_color=None,
            items=[],
            view=gws.gis.render.view_from_bbox(
                crs=rd.req.kparam('crs') or rd.req.kparam('srs') or rd.project.map.crs,
                bbox=bbox,
                out_size=(px_width, px_height),
                out_size_unit='px',
                rotation=0,
                dpi=0)
        )

        for node in nodes:
            render_input.items.append(t.RenderInputItem(
                type=t.RenderInputItemType.image_layer,
                layer=node.layer))

        renderer = gws.gis.render.Renderer()
        for _ in renderer.run(render_input):
            pass

        out = renderer.output
        if not out.items:
            img = gws.tools.misc.Pixels.png8
        else:
            buf = io.BytesIO()
            out.items[0].image.save(buf, format='png')
            img = buf.getvalue()

        return t.HttpResponse({
            'mime': 'image/png',
            'content': img
        })

    def handle_getlegendgraphic(self, rd: ows.OwsRequest):
        # https://docs.geoserver.org/stable/en/user/services/wms/get_legend_graphic/index.html
        # @TODO currently only support 'layer'

        nodes = self.layer_nodes_from_request_params(rd, 'layers', 'layer')
        if not nodes:
            raise gws.web.error.NotFound()

        img = nodes[0].layer.render_legend()

        return t.HttpResponse({
            'mime': 'image/png',
            'content': img or gws.tools.misc.Pixels.png8
        })

    def handle_getfeatureinfo(self, rd: ows.OwsRequest):
        results = self.find_features(rd)
        nodes = self.feature_node_list(rd, results)
        return self.render_feature_nodes(rd, nodes, 'getFeatureInfo')

    def find_features(self, rd: ows.OwsRequest):
        try:
            bbox = [float(n) for n in rd.req.kparam('bbox').split(',')]
            px_width = int(rd.req.kparam('width'))
            px_height = int(rd.req.kparam('height'))
            limit = int(rd.req.kparam('feature_count', '1'))
            i = int(rd.req.kparam('i'))
            j = int(rd.req.kparam('j'))
        except:
            raise gws.web.error.BadRequest()

        nodes = self.layer_nodes_from_request_params(rd, 'query_layers')
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
