import os
import re
import datetime
import pytest

import wand.image
import requests
from lxml import etree

import gws

import gws.ext.db.provider.postgres
import gws.gis.feature
import gws.gis.proj
import gws.tools.json2
import gws.tools.xml2
import gws.tools.os2
import gws.tools.vendor.umsgpack as umsgpack

import gws.types as t

DIR = os.path.dirname(__file__)


def raises(exc):
    """Shortcut for pytest.raises."""

    return pytest.raises(exc)


def read(path, mode='rt'):
    """Read a file and return its content."""

    try:
        with open(path, mode) as fp:
            return fp.read()
    except Exception as e:
        return f'FILE ERROR: {e}, path {path!r}'


def json(src):
    """Encode `src` as pretty json."""

    return gws.tools.json2.to_pretty_string(src)


_replace_attributes = {
    'timeStamp': '...',
}


def xml(src, save_to=None):
    """Format an xml document/file/response for better diffing."""

    if hasattr(src, 'text'):
        text = src.text
    elif src.endswith('.xml'):
        text = read(src)
    else:
        text = src

    text = gws.as_str(text)
    if save_to:
        with open(save_to, 'wt') as fp:
            fp.write(text)

    def as_str(el, indent):
        i1 = '    ' * indent
        i2 = '    ' * (indent + 1)

        s = i1 + '<' + el.qname
        for k, v in sorted(el.attr_dict.items()):
            s += '\n' + i2 + k + '="' + gws.tools.xml2.encode(_replace_attributes.get(k, v)) + '"'
        if not el.text and not el.children:
            s += '/>'
            return s
        s += '>'
        if el.text:
            s += '\n' + i2 + re.sub(r'\s+', ' ', el.text.strip())
        for c in el.children:
            s += '\n' + as_str(c, indent + 1)
        s += '\n' + i1 + '</' + el.qname + '>'
        return s

    try:
        root = gws.tools.xml2.from_string(text)
    except Exception as e:
        return f'INVALID XML:\n{e}\nRAW CONTENT :\n{text}\nFROM {src!r}'
    return as_str(root, 0)


def strlist(ls):
    """Format a list as a comma-separated string."""

    return ','.join(str(p) for p in ls)


def strlines(s):
    """Strip each line in a text"""

    return '\n'.join(gws.lines(s))


_LOCALHOST = 'http://127.0.0.1'


def req(url, **kwargs) -> requests.Response:
    """Perform a get request to the local server."""

    url = _LOCALHOST + '/' + url
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
        url=_LOCALHOST + '/_',
        method='POST',
        data=data,
        headers=headers,
        **kwargs)

    return res


def compare_image(r: requests.Response, path, threshold=0.00001):
    """Compare an image in the request with a reference image."""

    def _cmp(content):
        if not os.path.exists(path):
            return f'IMAGE ERROR: {path}: missing'

        a = wand.image.Image(blob=content)
        b = wand.image.Image(filename=path)

        if a.width != b.width:
            return f'IMAGE ERROR: {path}: bad width {a.width} != {b.width}'
        if a.height != b.height:
            return f'IMAGE ERROR: {path}: bad height {a.height} != {b.height}'

        _, diff = a.compare(b, metric='mean_absolute')

        if diff > threshold:
            return f'IMAGE ERROR: {path}: too different diff={diff:f} threshold={threshold:f}'

    if isinstance(r, str):
        return 'NO ERROR', r

    d = gws.ensure_dir(gws.VAR_DIR + '/response_images')
    with open(d + '/' + path.split('/')[-1], 'wb') as fp:
        fp.write(r.content)

    err = _cmp(r.content)
    if not err:
        return True, True
    return 'NO ERROR', err


def compare_xml(r: requests.Response, content=None, path=None):
    a = xml(r.text)
    b = xml(content or path)
    if a != b:
        p = path or (gws.sha256(content) + '.xml')
        d = gws.ensure_dir(gws.VAR_DIR + '/response_xml')
        with open(d + '/' + p.split('/')[-1], 'wt') as fp:
            fp.write(r.text)
    return a, b


def short_features(fs, trace=False):
    """Pick essential properties from a list of features."""

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


def postgres_select(stmt):
    """Perform a SELECT in postgres and return records as a list of dicts."""

    with _postgres_provider().connect() as conn:
        return list(conn.select(stmt))


def make_features(target, geom_type, prop_schema, crs, xy, rows, cols, gap):
    """Generate features on a rectangular grid in a postgres table or a geojson file.

    Args:
        target: Either 'postgres:<table name>' or '<path>.geojson'
        geom_type: One of 'point' or 'square'
        prop_schema: A dictionary {property_name: type} where type is one of 'int', 'float', 'text', 'date'
        crs: A CRS like 'EPGS:3857'
        xy: Start point
        rows: Number of rows
        cols: Number of columns
        gap: Distance between features in projection units
    """

    if target.startswith('postgres:'):
        name = target.split(':')[1]
        features = _make_geom_features(name, geom_type, prop_schema, crs, xy, rows, cols, gap)
        prov = _postgres_provider()
        gt = geom_type
        if geom_type == 'square':
            gt = 'polygon'
        _postgres_create_table(prov, name, prop_schema, gt, crs)
        table = prov.configure_table({'name': name})
        prov.edit_operation('insert', table, features)
        next_id = max(int(f.uid) for f in features) + 1
        with prov.connect() as conn:
            conn.execute(f"ALTER SEQUENCE {name}_id_seq RESTART WITH {next_id}")
    else:
        name = gws.tools.os2.parse_path(target)['name']
        features = _make_geom_features(name, geom_type, prop_schema, crs, xy, rows, cols, gap)
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

        gws.tools.json2.to_path(target, js, pretty=True)


##


_postgres = None


def _postgres_provider() -> gws.ext.db.provider.postgres.Object:
    global _postgres

    if _postgres:
        return _postgres

    cfg = _test_config()
    conn = t.Data(
        host=cfg['host.name'],
        port=cfg['postgres_connection.port'],
        user=cfg['postgres_connection.user'],
        password=cfg['postgres_connection.password'],
        database=cfg['postgres_connection.database'],
    )
    _postgres = gws.ext.db.provider.postgres.Object()
    _postgres.initialize(conn)
    return _postgres


def _postgres_create_table(prov, name, prop_schema, geom_type, crs):
    props = ','.join(f'{k} {v}' for k, v in prop_schema.items())
    srid = crs.split(':')[-1]

    ddl = f'''
        CREATE TABLE {name} (
            id SERIAL PRIMARY KEY,
            {props},
            p_geom GEOMETRY({geom_type},{srid})
        )
    '''
    with prov.connect() as conn:
        conn.execute(f'DROP TABLE IF EXISTS {name}')
        conn.execute(ddl)


def _test_config():
    # see cmd.py/start_container
    return gws.tools.json2.from_path(gws.VAR_DIR + '/cmd.ini.json')


def _make_geom_features(name, geom_type, prop_schema, crs, xy, rows, cols, gap):
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

            features.append(gws.gis.feature.Feature(
                uid=uid,
                attributes=atts,
                shape={'crs': crs, 'geometry': geom}
            ))

    return features
