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

    def handle_getcapabilities(self, rd: ows.Request):
        root = self.layer_tree_root(rd)
        if not root:
            raise gws.web.error.NotFound()
        return self.xml_response(self.render_template(rd, 'getCapabilities', {
            'layer_tree_root': root,
        }))

    def handle_getmap(self, rd: ows.Request):
        try:
            bbox = gws.gis.extent.from_string(rd.req.param('bbox'))
            px_width = int(rd.req.param('width'))
            px_height = int(rd.req.param('height'))
        except:
            raise gws.web.error.BadRequest()

        if not bbox or not px_width or not px_height:
            raise gws.web.error.BadRequest()

        nodes = self.layer_nodes_from_request_params(rd, ['layer', 'layers'])
        if not nodes:
            raise gws.web.error.NotFound()

        render_input = t.RenderInput(
            background_color=None,
            items=[],
            view=gws.gis.render.view_from_bbox(
                crs=rd.req.param('crs') or rd.req.param('srs') or rd.project.map.crs,
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

    def handle_getlegendgraphic(self, rd: ows.Request):
        # https://docs.geoserver.org/stable/en/user/services/wms/get_legend_graphic/index.html
        # @TODO currently only support 'layer'

        nodes = self.layer_nodes_from_request_params(rd, ['layer', 'layers'])
        if not nodes:
            raise gws.web.error.NotFound()

        img = nodes[0].layer.render_legend()

        return t.HttpResponse({
            'mime': 'image/png',
            'content': img or gws.tools.misc.Pixels.png8
        })

    def handle_getfeatureinfo(self, rd: ows.Request):
        results = self.find_features(rd)
        nodes = self.feature_node_list(rd, results)
        return self.render_feature_nodes(rd, nodes, 'getFeatureInfo')

    def find_features(self, rd: ows.Request):
        try:
            px_width = int(rd.req.param('width'))
            px_height = int(rd.req.param('height'))
            limit = int(rd.req.param('feature_count', '1'))
            x = int(rd.req.param('i') or rd.req.param('x'))
            y = int(rd.req.param('j') or rd.req.param('y'))
        except:
            raise gws.web.error.BadRequest()

        bbox = gws.gis.extent.from_string(rd.req.param('bbox'))
        crs = rd.req.param('crs') or rd.req.param('srs')

        nodes = self.layer_nodes_from_request_params(rd, ['query_layers'])
        if not nodes:
            raise gws.web.error.NotFound()

        xres = (bbox[2] - bbox[0]) / px_width
        yres = (bbox[3] - bbox[1]) / px_height
        x = bbox[0] + (x * xres)
        y = bbox[3] - (y * yres)

        point = gws.gis.shape.from_props(t.ShapeProps(
            crs=rd.project.map.crs,
            geometry={
                'type': 'Point',
                'coordinates': [x, y]
            }
        ))

        # @TODO: should be a parameter
        pixel_tolerance = 10

        args = t.SearchArgs(
            project=rd.project,
            layers=[n.layer for n in nodes],
            limit=min(limit, MAX_LIMIT),
            resolution=xres,
            shapes=[point],
            tolerance=(pixel_tolerance, 'px'),
        )

        return gws.common.search.runner.run(rd.req, args)
