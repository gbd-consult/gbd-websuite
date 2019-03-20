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
        return util.xml_response(tag(
            'WMS_Capabilities',
            self.caps_service(),
            self.caps_capability(),
            {
                'version': VERSION,
                'xmlns': 'http://www.opengis.net/wms',
                'xmlns:xlink': 'http://www.w3.org/1999/xlink',
                'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                'xsi:schemaLocation': 'http://www.opengis.net/wms http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd',
            }
        ))

    def caps_service(self):
        return tag(
            'Service',
            tag('Name', 'WMS'),
            tag('Title', self.project.title),
            tag('Abstract', self.project.meta.abstract),
        )

    def caps_capability(self):
        return tag(
            'Capability',
            self.caps_request_tag(),
            tag('Exception', tag('Format', 'XML')),
            self.caps_top_layer(),
        )

    def caps_request_tag(self):
        url = self.req.reversed_url(f'cmd=owsHttpGet&projectUid={self.project.uid}')

        link = tag('DCPType', tag('HTTP', tag('Get', tag('OnlineResource', {
            'xmlns:xlink': 'http://www.w3.org/1999/xlink',
            'xlink:type': 'simple',
            'xlink:href': url
        }))))

        return tag(
            'Request',
            tag('GetCapabilities', tag('Format', 'text/xml'), link),
            tag('GetMap', tag('Format', 'image/png'), link),
            tag('GetFeatureInfo', tag('Format', 'application/vnd.ogc.gml'), link),
        )

    def caps_top_layer(self):
        map = self.project.map
        return tag(
            'Layer',
            tag('Name', self.project.uid),
            tag('Title', self.project.title),
            tag('CRS', map.crs),
            _geographic_bounding_box(map.extent, map.crs),
            _bounding_box(map.extent, map.crs),
            *gws.compact(self.caps_layer(la) for la in map.layers)
        )

    def caps_layer(self, layer: t.LayerObject):
        sub_layers = []

        if layer.layers:
            sub_layers = gws.compact(self.caps_layer(la) for la in layer.layers)
            if not sub_layers:
                return
        elif not layer.is_enabled_for_service('wms'):
            return

        return tag(
            'Layer',
            {'queryable': 1 if layer.has_search else 0},
            tag('Name', layer.uid),
            tag('Title', layer.title),
            tag('Abstract', layer.meta.abstract),
            tag('CRS', layer.map.crs),
            *sub_layers
        )

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
            layers.extend(self.collect_wms_layers(layer_uid))

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

    def collect_wms_layers(self, layer_uid):
        la = self.req.require('gws.ext.layer', layer_uid)

        if la and la.layers:
            ls = []
            for sub in la.layers:
                ls.extend(self.collect_wms_layers(sub.uid))
            return ls

        if la.is_enabled_for_service('wms'):
            return [la]

        return []

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
            layers.extend(self.collect_wms_layers(layer_uid))

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


def _geographic_bounding_box(extent, crs):
    e = gws.gis.proj.transform_bbox(extent, crs, 'EPSG:4326')
    return tag(
        'EX_GeographicBoundingBox',
        tag('westBoundLongitude', '%.3f' % e[0]),
        tag('eastBoundLongitude', '%.3f' % e[2]),
        tag('southBoundLatitude', '%.3f' % e[1]),
        tag('northBoundLatitude', '%.3f' % e[3]),
    )


def _bounding_box(extent, crs):
    return tag('BoundingBox', {
        'CRS': crs,
        'minx': extent[0],
        'miny': extent[1],
        'maxx': extent[2],
        'maxy': extent[3],
    })


def _keyword_tag(self, keywords):
    if keywords:
        return {
            'tag': 'KeywordList',
            'children': [
                {'tag': 'Keyword', 'text': k}
                for k in keywords
            ]
        }


def _as_ident(s):
    return re.sub(r'\W+', '_', s)