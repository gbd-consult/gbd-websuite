import re

import gws.common.search.runner
import gws.gis.proj
import gws.gis.render
import gws.gis.shape
import gws.ows.gml
import gws.tools.misc as misc
import gws.tools.shell
import gws.tools.xml3
import gws.types as t
import gws.web.error
from . import util

tag = gws.tools.xml3.tag

VERSION = '1.3.0'
MAX_LIMIT = 100


def request(action, req, params):
    try:
        project = req.require_project(params.get('projectuid'))

        if params.get('version') and params.get('version') != VERSION:
            raise gws.web.error.NotFound()

        writer = WmsWriter(action, project, req, params)
        r = params.get('request', '').lower()

        if r == 'getcapabilities':
            return writer.getcapabilities()
        if r == 'getmap':
            return writer.getmap()
        if r == 'getfeatureinfo':
            return writer.getfeatureinfo()

        raise gws.web.error.NotFound()

    except gws.web.error.HTTPException as err:
        return util.xml_exception_response(VERSION, err.code, err.description)
    except:
        gws.log.exception()
        return util.xml_exception_response(VERSION, '500', 'Internal Server Error')


class WmsWriter:
    def __init__(self, action, project, req, params):
        self.action = action
        self.project = project
        self.req = req
        self.params = params

    def getcapabilities(self):
        context = {
            'project': self.project,
            #'writer': self,
            'layer_tree': self.layer_tree(),
            'service_endpoint': self.req.reversed_url(f'cmd=owsHttpGet&projectUid={self.project.uid}')

        }
        tpl = self.action.templates.get('wmsGetCapabilities')
        out = tpl.render(context)
        return util.xml_response(out.content.strip())

    ##

    def getmap(self):
        try:
            layer_uids = self.params.get('layers').split(',')
            bbox = [float(n) for n in self.params.get('bbox').split(',')]
            px_width = int(self.params.get('width'))
            px_height = int(self.params.get('height'))
        except:
            raise gws.web.error.BadRequest()

        layers = []
        for layer_uid in layer_uids:
            layers.extend(self.layer_list(layer_uid))

        if not layers:
            raise gws.web.error.NotFound()

        render_input = t.MapRenderInput({
            'out_path': '/tmp/' + gws.random_string(64) + '.png',
            'bbox': bbox,
            'rotation': 0,
            'scale': misc.res2scale((bbox[2] - bbox[0]) / px_width),
            'dpi': 0,
            'map_size_px': [px_width, px_height],
            'items': [],
        })

        for la in layers:
            item = t.MapRenderInputItem({
                'layer': la,
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

    ##

    def getfeatureinfo(self):
        try:
            layer_uids = self.params.get('query_layers').split(',')
            bbox = [float(n) for n in self.params.get('bbox').split(',')]
            px_width = int(self.params.get('width'))
            px_height = int(self.params.get('height'))
            limit = int(self.params.get('feature_count', '1'))
            i = int(self.params.get('i'))
            j = int(self.params.get('j'))
        except:
            raise gws.web.error.BadRequest()

        layers = []
        for layer_uid in layer_uids:
            layers.extend(self.layer_list(layer_uid))

        if not layers:
            raise gws.web.error.NotFound()

        xres = (bbox[2] - bbox[0]) / px_width
        yres = (bbox[3] - bbox[1]) / px_height
        x = bbox[0] + (i * xres)
        y = bbox[3] - (j * yres)

        gws.p('WMS_POINT', bbox, i, j, xres, yres, x, y)

        point = gws.gis.shape.from_props(t.ShapeProps({
            'crs': self.project.map.crs,
            'geometry': {
                'type': 'Point',
                'coordinates': [x, y]
            }}
        ))

        self.pixel_tolerance = 10

        args = t.SearchArgs({
            'bbox': bbox,
            'crs': self.project.map.crs,
            'project': None,
            'keyword': None,
            'layers': layers,
            'limit': min(limit, MAX_LIMIT),
            'resolution': xres,
            'shapes': [point],
            'tolerance': self.pixel_tolerance * xres,
        })

        features = gws.common.search.runner.run(self.req, args)

        return util.xml_response(tag(
            'wfs:FeatureCollection',
            {
                'version': VERSION,
                'xmlns': 'http://www.opengis.net/wfs',
                'xmlns:wfs': 'http://www.opengis.net/wfs/2.0',
                'xmlns:gml': 'http://www.opengis.net/gml/3.2',
                'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                'xsi:schemaLocation': 'http://www.opengis.net/wfs/2.0 http://schemas.opengis.net/wfs/2.0/wfs.xsd',
            },
            self.gml_bounded_by(features),
            *gws.compact(self.gml_feature(f) for f in features)
        ))

    def gml_bounded_by(self, features):
        return ''

    def gml_feature(self, f: t.FeatureInterface):
        props = [tag(_as_ident(k), v) for k, v in f.attributes.items()]
        if f.shape:
            props.append(tag(
                'geometry',
                gws.ows.gml.shape_to_tag(f.shape, precision=self.project.map.coordinate_precision)))
        return tag('wfs:member', tag(f.category, {'gml:id': f.uid}, *props))

    ##

    def layer_list(self, layer_uid):
        layer = self.req.acquire('gws.ext.layer', layer_uid)

        if not layer or not layer.is_enabled_for_service('wms'):
            return []

        if layer.layers:
            ls = []
            for la in layer.layers:
                ls.extend(self.layer_list(la.uid))
            return ls

        return [layer]

    def layer_subtree(self, layer_uid):
        layer: t.LayerObject = self.req.acquire('gws.ext.layer', layer_uid)

        if not layer or not layer.is_enabled_for_service('wms'):
            return

        sub = []

        if layer.layers:
            sub = gws.compact(self.layer_subtree(la.uid) for la in layer.layers)
            if not sub:
                return

        res = [misc.res2scale(r) for r in layer.resolutions]
        crs = layer.map.crs

        return {
            'layer': layer,
            'extent': layer.extent,
            'epsg4326extent': gws.gis.proj.transform_bbox(layer.extent, crs, 'EPSG:4326'),
            'crs': crs,
            'queryable': layer.has_search or any(s['queryable'] for s in sub),
            'min_scale': min(res),
            'max_scale': max(res),
            'sub_nodes': list(reversed(sub)),  # NB: WMS is bottom-first, our layers are top-first
        }

    def layer_tree(self):
        sub = gws.compact(self.layer_subtree(la.uid) for la in self.project.map.layers)
        return list(reversed(sub))


def _as_ident(s):
    return re.sub(r'\W+', '_', s)
