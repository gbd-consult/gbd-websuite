import os
import re
import datetime
import itertools

import wand.image
import requests
from lxml import etree

import gws

import gws.ext.db.provider.postgres
import gws.gis.feature
import gws.gis.proj
import gws.tools.json2
import gws.tools.misc
import gws.tools.vendor.umsgpack as umsgpack

import gws.types as t

DIR = os.path.dirname(__file__)


def read(path):
    try:
        with open(path) as fp:
            return fp.read()
    except Exception as e:
        return f'FILE ERROR: {e}'


def print_json(x):
    print('-' * 40)
    print(gws.tools.json2.to_pretty_string(x))
    print('-' * 40)


def xml(x):
    try:
        xml = etree.fromstring(x.encode('utf8'))
    except Exception as e:
        return f'INVALID XML:\n{e}\nRAW CONTENT :\n{x}'
    return etree.tounicode(xml, pretty_print=True)


def print_xml(x):
    print('-' * 40)
    print(xml(x))
    print('-' * 40)


def strlist(ls):
    return ','.join(str(p) for p in ls)


def req(url, **kwargs) -> requests.Response:
    """Perform a get request to the local server."""

    url = 'http://127.0.0.1/' + url
    if 'params' in kwargs:
        qs = gws.as_query_string(kwargs.pop('params'))
        b = '&' if '?' in url else '?'
        url += b + qs

    res = requests.request(
        url=url,
        method='GET',
        **kwargs)

    return res


def cmd(command, params=None, binary=False, **kwargs):
    """Perform an api command on the local server."""

    p = {
        'cmd': command,
        'params': params
    }
    if binary:
        data = umsgpack.dumps(p)
        ct = 'application/msgpack'
    else:
        data = gws.tools.json2.to_string(p)
        ct = 'application/json'

    headers = {
        'Content-type': ct,
        'Accept': 'application/json',
    }

    res = requests.request(
        url='http://127.0.0.1/_',
        method='POST',
        data=data,
        headers=headers,
        **kwargs)

    return res


def test_config():
    # see cmd.py/start_container
    return gws.tools.json2.from_path(gws.VAR_DIR + '/test.config.json')


def compare_image_response(r: requests.Response, path, threshold=0.1):
    def _cmp():
        if not os.path.exists(path):
            return f'missing {path}'

        a = wand.image.Image(blob=content)
        b = wand.image.Image(filename=path)

        if a.width != b.width:
            return f'bad width {a.width} != {b.width}'
        if a.height != b.height:
            return f'bad height {a.height} != {b.height}'

        loc, diff = a.similarity(b)

        if loc['left'] != 0 or loc['top'] != 0 or loc['width'] != b.width or loc['height'] != b.height:
            return f'bad loc {loc}'

        if diff > threshold:
            return f'bad diff {diff} > {threshold}'

        return ''

    content = r.content
    err = _cmp()
    if err:
        name = path.split('/')[-1]
        d = gws.tools.misc.ensure_dir(gws.VAR_DIR + '/failed_files')
        with open(d + '/' + name, 'wb') as fp:
            fp.write(content)
    return err


def short_features(fs, trace=False):
    rs = []
    for f in fs:
        r = {}
        if f.get('uid'):
            r['uid'] = f['uid']
        if f.get('attributes'):
            r['attributes'] = ' '.join(f"{a['name']}=<{a['value']}>" for a in f['attributes'])
        if f.get('shape'):
            r['geometry'] = f['shape']['geometry']['type'].upper() + ' ' + f['shape']['crs']
        rs.append(r)
    if trace:
        print('-' * 40)
        print(gws.tools.json2.to_string(rs, pretty=True))
        print('-' * 40)
    return rs


def postgres_provider():
    cfg = test_config()['postgres_container']['connection']
    prov = gws.ext.db.provider.postgres.Object()
    prov.initialize(t.Data(cfg))
    return prov


def make_geom_features(name, geom_type, prop_schema, crs, xy, rows, cols, gap):
    """Generate features with on a rectangular grid of points"""

    features = []

    sx, sy = gws.gis.proj.transform_xy(xy[0], xy[1], 'EPSG:3857', crs)

    for r in range(rows):
        for c in range(cols):
            uid = r * cols + (c + 1)

            atts = {}
            for k, v in prop_schema.items():
                if v == 'int':
                    atts[k] = uid * 100
                if v == 'float':
                    atts[k] = uid * 200.0
                if v == 'text':
                    atts[k] = f"{name}/{uid}"
                if v == 'date':
                    atts[k] = datetime.datetime(2019, 1, 1) + datetime.timedelta(days=uid - 1)

            x = sx + c * gap
            y = sy + r * gap

            if geom_type == 'point':
                geom = {
                    'type': 'Point',
                    'coordinates': [x, y]
                }

            if geom_type == 'square':
                w = h = gap / 2
                geom = {
                    'type': 'Polygon',
                    'coordinates': [[
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h],
                        [x, y],
                    ]]
                }

            if geom_type == 'polygon':
                w = h = gap / 2
                geom = {
                    'type': 'Polygon',
                    'coordinates': [[
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h],
                        [x, y],
                    ]]
                }

            features.append(gws.gis.feature.Feature(
                uid=uid,
                attributes=atts,
                shape={'crs': crs, 'geometry': geom}
            ))

    return features


def create_postgis_table(prov, name, prop_schema, geom_type, crs):
    props = ','.join(f'{k} {v}' for k, v in prop_schema.items())
    srid = crs.split(':')[-1]

    ddl = f'''
        CREATE TABLE {name} (
            id INTEGER PRIMARY KEY,
            {props},
            p_geom GEOMETRY({geom_type},{srid})
        )
    '''
    with prov.connect() as drv:
        drv.execute(f'DROP TABLE IF EXISTS {name}')
        drv.execute(ddl)


def make_geom_table(name, geom_type, prop_schema, crs, xy, rows, cols, gap):
    prov = postgres_provider()
    gt = geom_type
    if geom_type == 'square':
        gt = 'polygon'
    create_postgis_table(prov, name, prop_schema, gt, crs)
    features = make_geom_features(name, geom_type, prop_schema, crs, xy, rows, cols, gap)
    table = prov.configure_table({'name': name})
    prov.edit_operation('insert', table, features)


def make_geom_json(path, geom_type, prop_schema, crs, xy, rows, cols, gap):
    features = make_geom_features(gws.tools.misc.parse_path(path)['name'], geom_type, prop_schema, crs, xy, rows, cols, gap)
    srid = crs.split(':')[-1]

    js = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::" + srid
            }
        },
        "features": []
    }

    for f in features:
        f = f.props
        props = {a.name: a.value for a in f.attributes}
        props['id'] = f.uid
        js["features"].append({
            "type": "Feature",
            "geometry": f.shape.geometry,
            "properties": props
        })

    gws.tools.json2.to_path(path, js, pretty=True)
