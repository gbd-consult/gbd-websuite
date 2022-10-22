"""Test web server.

This servers runs in a dedicated docker container during testing
and acts as a mock for our http-related functonality.

This server can process the following requests:

GET requests:

/<service-name>?params  - route the request to a preconfigured service
/<path-name>            - serve a file from the server directory
/<any string>           - serve a prededined response (see `poke` below)

POST requests (all in JSON format):

{ cmd: "poke", pattern: regex, response: PokeResponse }  - configure a predefined response for all GET requests matching `pattern`
{ cmd: "reset" } - reset the internal state:
{ cmd: "begin_capture" } - begin recording all incoming GET urls
{ cmd: "end_capture" } - end recording incoming urls, returns `{urls: [list of captured urls]}`:
{ cmd: "create_wms", config: WmsServiceConfig } - create a WMS service

"""

import http.server
import io
import json
import os
import re
import time
import urllib.parse

from PIL import Image, ImageColor, ImageDraw, ImageFont

from typing import List, Tuple

# ows services

Point = Tuple[float, float]
Extent = Tuple[float, float, float, float]


class WmsLayerConfig:
    name: str
    extent: Extent
    color: str
    queryable: bool
    parent: str
    points: List[Point]


class WmsLayer:
    def __init__(self, config):
        self.name = config['name']

        ext = config.get('extent')
        if ext:
            self.extent = parse_box(ext)
            self.width = self.extent[2] - self.extent[0]
            self.height = self.extent[3] - self.extent[1]
        else:
            self.extent = []
            self.width = self.height = 0

        self.color = config.get('color')

        self.queryable = config.get('queryable', False)
        self.parent = config.get('parent')

        self.points = config.get('points', [])

    def __repr__(self):
        return repr(vars(self))

    def feature_at(self, service, cx, cy):
        for p in self.points:
            if p[0] == cx and p[1] == cy:
                return {'name': self.name, 'cx': cx, 'cy': cy}

    def wms_caps(self, service):
        crs = 'CRS' if '3' in service.version else 'SRS'
        return tag(
            'Layer',
            {'queryable': '1' if self.queryable else '0'},
            tag('Name', f'{self.name}'),
            tag('Title', f'{self.name}_TITLE'),
            tag('Abstract', f'{self.name}_ABSTRACT'),
            tag('KeywordList',
                tag('Keyword', f'{self.name}_KEYWORD_1'),
                tag('Keyword', f'{self.name}_KEYWORD_2'),
                tag('Keyword', f'{self.name}_KEYWORD_3')),
            tag(crs, service.crs),
            self.extent and tag(
                'BoundingBox',
                {
                    crs: service.crs,
                    'minx': self.extent[0],
                    'miny': self.extent[1],
                    'maxx': self.extent[2],
                    'maxy': self.extent[3],
                }),
            [la.wms_caps(service) for la in service.layers if la.parent == self.name]
        )


class WmsServiceConfig:
    version: str
    url: str
    crs: str
    layers: List[WmsLayerConfig]


class WmsService:
    def __init__(self, name, config):
        self.name = name
        self.version = config.get('version', '1.3.0')
        self.url = config['url']
        self.crs = config.get('crs', 'EPSG:3857')
        self.layers = [WmsLayer(c) for c in config.get('layers', [])]

    def __repr__(self):
        return repr(vars(self))

    def handle(self, params):
        if 'bbox' in params:
            params['bbox'] = parse_box(params['bbox'])

        ls = params.get('layers')
        params['layers'] = [la for la in self.layers if la.name in ls.split(',')] if ls else self.layers

        ls = params.get('query_layers')
        params['query_layers'] = [la for la in self.layers if la.name in ls.split(',')] if ls else self.layers

        params['width'] = intf(params.get('width'))
        params['height'] = intf(params.get('height'))

        params['x'] = intf(params.get('x') or params.get('i'))
        params['y'] = intf(params.get('y') or params.get('j'))

        fn = 'handle_' + params.get('request').lower()
        return getattr(self, fn)(params)

    def handle_getcapabilities(self, params):
        root_layer = [la for la in self.layers if not la.parent][0]

        def request_caps(verb, format):
            return tag(verb,
                       tag('Format', format),
                       tag('DCPType', tag('HTTP', tag('Get', tag('OnlineResource', {'xlink:href': self.url})))))

        return xml_response(
            tag('WMS_Capabilities',
                {'xmlns:xlink': "http://www.w3.org/1999/xlink"},
                {'version': self.version},
                tag('Service',
                    tag('Name', f'{self.name}'),
                    tag('Title', f'{self.name}_TITLE'),
                    tag('Abstract', f'{self.name}_ABSTRACT'),
                    tag('KeywordList',
                        tag('Keyword', f'{self.name}_KEYWORD_1'),
                        tag('Keyword', f'{self.name}_KEYWORD_2'),
                        tag('Keyword', f'{self.name}_KEYWORD_3')),
                    tag('OnlineResource', {'xlink:href': f'http://{self.name}_ONLINERESOURCE'}),
                    tag('ContactInformation',
                        tag('ContactPersonPrimary',
                            tag('ContactPerson', f'{self.name}_CONTACTPERSON'),
                            tag('ContactOrganization', f'{self.name}_CONTACTORGANIZATION')),
                        tag('ContactPosition', f'{self.name}_CONTACTPOSITION'),
                        tag('ContactAddress',
                            tag('AddressType', f'{self.name}_ADDRESSTYPE'),
                            tag('Address', f'{self.name}_ADDRESS'),
                            tag('City', f'{self.name}_CITY'),
                            tag('StateOrProvince', f'{self.name}_STATEORPROVINCE'),
                            tag('PostCode', f'{self.name}_POSTCODE'),
                            tag('Country', f'{self.name}_COUNTRY')),
                        tag('ContactVoiceTelephone', f'{self.name}_CONTACTVOICETELEPHONE'),
                        tag('ContactFacsimileTelephone', f'{self.name}_CONTACTFACSIMILETELEPHONE'),
                        tag('ContactElectronicMailAddress', f'{self.name}_CONTACTELECTRONICMAILADDRESS')),
                    tag('Fees', f'{self.name}_FEES'),
                    tag('AccessConstraints', f'{self.name}_ACCESSCONSTRAINTS'),
                    tag('MaxWidth', 2048),
                    tag('MaxHeight', 2048)),
                tag('Capability',
                    tag('Request',
                        request_caps('GetCapabilities', 'text/xml'),
                        request_caps('GetMap', 'image/png'),
                        request_caps('GetFeatureInfo', 'text/xml')),
                    tag('Exception', tag('Format', 'XML')),
                    root_layer.wms_caps(self))
                ))

    def handle_getmap(self, params):
        return 'image/png', draw_dict(params, params['width'], params['height'])

    def handle_getfeatureinfo(self, params):
        features = []
        for la in params['query_layers']:
            if la.queryable:
                f = la.feature_at(self, params['x'], params['y'])
                if f:
                    features.append(f)

        return xml_response(
            tag('wfs:FeatureCollection',
                {
                    'xmlns:xs': "http://www.w3.org/2001/XMLSchema",
                    'xmlns:wfs': "http://www.opengis.net/wfs/2.0",
                    'xmlns:gml': "http://www.opengis.net/gml/3.2",
                    'xmlns:xlink': "http://www.w3.org/1999/xlink",
                    'numberMatched': "unknown",
                    'numberReturned': len(features),
                },
                tag('wfs:member', [
                    tag('test:feature',
                        {'id': 1},
                        tag('test:title', f['name']),
                        tag('test:cx', f['cx']),
                        tag('test:cy', f['cy']),
                        tag('gml:Point', {'id': 1}, tag('gml:pos', str(f['cx']) + ' ' + str(f['cy'])))
                        ) for f in features]
                    )))


# utils

def xml_response(doc):
    return 'text/xml', '<?xml version="1.0" encoding="UTF-8"?>\n' + doc


def tag(name, *args):
    def encode(v: str) -> str:
        v = str(v).strip()
        v = v.replace("&", "&amp;")
        v = v.replace(">", "&gt;")
        v = v.replace("<", "&lt;")
        v = v.replace('"', "&quot;")
        return v

    atts = {}
    sub = []

    for a in args:
        if not a:
            continue
        if isinstance(a, dict):
            atts.update(a)
            continue
        if not isinstance(a, list):
            a = [a]
        for t in a:
            t = str(t)
            if not t.startswith('<'):
                t = encode(t)
            sub.append(t)

    res = name
    atts_str = ''

    if atts:
        atts_str = ' '.join(f'{k}="{encode(v)}"' for k, v in atts.items() if v is not None)
    if atts_str:
        res += ' ' + atts_str
    if sub:
        return '<' + res + '>' + ''.join(sub) + '</' + name + '>'
    return '<' + res + '/>'


def image_from_bytes(r):
    return Image.open(io.BytesIO(r))


def image_to_bytes(img):
    buf = io.BytesIO()
    img.tobytes()
    img.save(buf, 'PNG')
    return buf.getvalue()


def draw_dict(d, width, height):
    img = Image.new('RGBA', (width, height), ImageColor.getrgb('#cccccc'))
    text = '\n'.join(str(k) + ':' + str(v) for k, v in sorted(d.items()))
    ImageDraw.Draw(img).multiline_text(
        (1, 1),
        text.strip(),
        spacing=0,
        font=ImageFont.load_default(),
        fill=ImageColor.getrgb('#000000'))
    return image_to_bytes(img)


def parse_box(s):
    if not isinstance(s, list):
        s = s.split(',')
    return [intf(t) for t in s]


def intf(s):
    try:
        return int(float(s))
    except:
        return 0


def log(*args):
    print('TWS>', *[repr(a) for a in args])


# server main

STATE = {
    'host': '0.0.0.0',
    'port': 8080,
    'cdir': os.path.dirname(__file__),
    'poke_responses': {},
    'services': {},
    'capture': [],
    'capturing': False,
}


class PokeResponse:
    delay_time: int = 0
    status_code: int = 200
    content_type: str = 'text/plain'
    text: str = ''
    headers: List[Tuple] = []


class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        if self.path == '/_state':
            return self.serve(200, 'text/plain', repr(STATE))

        if STATE['capturing']:
            STATE['capture'].append(self.path)

        if '?' in self.path:
            p = self.path.split('?', 1)
            base = p[0].strip('/')
            params = {k.lower(): v[0] for k, v in urllib.parse.parse_qs(p[1]).items()}
        else:
            base = self.path.strip('/')
            params = {}

        if base in STATE['services']:
            srv = STATE['services'][base]
            log(f'found service: {base!r} => {srv.name!r}')
            mime, res = srv.handle(params)
            return self.serve(200, mime, res)

        for pattern, res in STATE['poke_responses'].items():
            if res and re.search(pattern, base):
                log(f'found response: {base!r} => {pattern!r}')
                if res.get('delay_time'):
                    time.sleep(int(res['delay_time']))
                return self.serve(
                    res.get('status_code', 200),
                    res.get('content_type', 'text/plain'),
                    res.get('text', ''),
                    res.get('headers'))

        fname = STATE['cdir'] + '/' + base
        if os.path.isfile(fname) and '..' not in fname:
            log(f'found file: {base} => {fname!r}')
            with open(fname, 'rb') as fp:
                return self.serve(200, 'application/octet-stream', fp.read())

        self.serve(404, 'text/plain', 'not found\n')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post = json.loads(self.rfile.read(content_length))
        res = getattr(self, 'cmd_' + post['cmd'])(post)
        if not res:
            res = {'ok': 1}
        self.serve(200, 'application/json', json.dumps(res))

    def cmd_poke(self, post):
        STATE['poke_responses'][post['pattern']] = post['response']

    def cmd_reset(self, post):
        STATE['poke_responses'] = dict()
        STATE['services'] = dict()
        STATE['capture'] = []
        STATE['capturing'] = False

    def cmd_begin_capture(self, post):
        STATE['capture'] = []
        STATE['capturing'] = True

    def cmd_end_capture(self, post):
        ls = list(STATE['capture'] or [])
        STATE['capture'] = []
        STATE['capturing'] = False
        return {'urls': ls}

    def cmd_create_wms(self, post):
        config = post['config']
        name = urllib.parse.urlsplit(config['url']).path.strip('/')
        STATE['services'][name] = WmsService(name, config)

    def serve(self, status, mime, text, headers=None):
        if isinstance(text, str):
            text = text.encode('utf8')
        self.send_response(status)
        h = {
            'content-length': str(len(text)),
            'content-type': mime,
        }
        if headers:
            for k, v in headers.items():
                h[k.lower()] = v
        for k, v in h.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(text)


def main():
    httpd = http.server.ThreadingHTTPServer((STATE['host'], STATE['port']), HTTPRequestHandler)
    httpd.serve_forever()


if __name__ == '__main__':
    main()
