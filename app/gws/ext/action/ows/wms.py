import gws.web.error
import gws.tools.xml3
import gws.gis.proj
import gws.tools.misc as misc
import gws.tools.shell
import gws.gis.render
import gws.types as t

from . import util

VERSION = '1.3.0'


def request(action, req, ps):
    try:
        project = req.require_project(ps.get('projectuid'))

        if ps.get('version') and ps.get('version') != VERSION:
            raise gws.web.error.NotFound()

        p = util.RequestParams({
            'action': action,
            'project': project,
            'req': req,
            'ps': ps
        })

        r = ps.get('request').lower()

        if r == 'getcapabilities':
            return _getcapabilities(p)
        if r == 'getmap':
            return _getmap(p)

        raise gws.web.error.NotFound()

    except gws.web.error.HTTPException as err:
        return util.xml_response(
            'ServiceExceptionReport',
            [
                ['ServiceException', err.description, {'code': err.code}]
            ],
            {
                'version': VERSION,
                'xmlns': 'http://www.opengis.net/ogc',
                'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                'xsi:schemaLocation': 'http://www.opengis.net/ogc http://schemas.opengis.net/wms/1.3.0/exceptions_1_3_0.xsd',
            },
        )


def _getcapabilities(p):
    return util.xml_response(
        'WMS_Capabilities',
        [
            _capabilities_service_tag(p),
            _capabilities_capability_tag(p),
        ],
        {
            'version': VERSION,
            'xmlns': 'http://www.opengis.net/wms',
            'xmlns:xlink': 'http://www.w3.org/1999/xlink',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://www.opengis.net/wms http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd',
        }
    )


def _capabilities_service_tag(p):
    return ['Service', [
        ['Name', 'WMS'],
        ['Title', p.project.title],
        ['Abstract', p.project.meta.abstract],
    ]]


def _capabilities_capability_tag(p):
    return ['Capability', [
        _capabilities_request_tag(p),
        ['Exception', [
            ['Format', 'XML']
        ]],
        _capabilities_top_layer_tag(p),
    ]]


def _capabilities_request_tag(p):
    url = p.req.reversed_url(f'cmd=owsHttpGet&projectUid={p.project.uid}')

    link = ['DCPType', [['HTTP', [['Get', [['OnlineResource', None, {
        'xmlns:xlink': 'http://www.w3.org/1999/xlink',
        'xlink:type': 'simple',
        'xlink:href': url
    }]]]]]]]

    return ['Request', [
        ['GetCapabilities', [
            ['Format', 'text/xml'],
            link
        ]],
        ['GetMap', [
            ['Format', 'image/png'],
            link
        ]],
        ['GetFeatureInfo', [
            ['Format', 'text/xml'],
            link
        ]]
    ]]


def _capabilities_top_layer_tag(p):
    map = p.project.map
    cn = [
        ['Name', p.project.uid],
        ['Title', p.project.title],
        ['CRS', map.crs],
        _geographic_bounding_box(map.extent, map.crs),
        _bounding_box(map.extent, map.crs),
    ]
    cn.extend(_capabilities_layer_tag(p, la) for la in map.layers)
    return ['Layer', cn]


def _capabilities_layer_tag(p, layer):
    gws.p(layer.uid)
    cn = [
        ['Name', layer.uid],
        ['Title', layer.title],
        ['Abstract', layer.meta.abstract],
        ['CRS', layer.map.crs],
        _geographic_bounding_box(layer.extent, layer.map.crs),
        _bounding_box(layer.extent, layer.map.crs),
    ]
    if hasattr(layer, 'layers'):
        cn.extend(_capabilities_layer_tag(p, la) for la in layer.layers)
    return ['Layer', cn]


def _geographic_bounding_box(extent, crs):
    e = gws.gis.proj.transform_bbox(extent, crs, 'EPSG:4326')
    return ['EX_GeographicBoundingBox', [
        ['westBoundLongitude', '%.3f' % e[0]],
        ['eastBoundLongitude', '%.3f' % e[1]],
        ['southBoundLatitude', '%.3f' % e[2]],
        ['northBoundLatitude', '%.3f' % e[3]],
    ]]


def _bounding_box(extent, crs):
    return ['BoundingBox', None, {
        'CRS': crs,
        'minx': extent[0],
        'miny': extent[1],
        'maxx': extent[2],
        'maxy': extent[3],
    }]


def _keyword_tag(self, keywords):
    if keywords:
        return {
            'tag': 'KeywordList',
            'children': [
                {'tag': 'Keyword', 'text': k}
                for k in keywords
            ]
        }


def _getmap(p):
    layers = []
    for layer_uid in p.ps.get('layers').split(','):
        layers.extend(_collect_wms_layers(p.req, layer_uid))

    if not layers:
        raise gws.web.error.NotFound()

    bbox = [float(n) for n in p.ps.get('bbox').split(',')]
    px_width = int(p.ps.get('width'))
    px_height = int(p.ps.get('height'))

    render_input = t.MapRenderInput({
        'out_path': '/tmp/' + gws.random_string(64) + '.png',
        'bbox': bbox,
        'rotation': 0,
        'scale': misc.res2scale((bbox[2] - bbox[0]) / px_width),
        'dpi': misc.OGC_SCREEN_PPI,
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


def _collect_wms_layers(req, layer_uid):
    la = req.require('gws.ext.layer', layer_uid)

    if la and la.layers:
        ls = []
        for sub in la.layers:
            ls.extend(_collect_wms_layers(req, sub.uid))
        return ls

    if la.is_enabled_for_service('wms'):
        return [la]

    return []
