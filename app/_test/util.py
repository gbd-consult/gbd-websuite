import os
import re
import datetime

import requests

import gws

import gws.tools.vendor.umsgpack as umsgpack
import gws.tools.json2
import gws.gis.feature

DIR = os.path.dirname(__file__)


def xml_nows(s):
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'> <', '><', s)
    return s


def xml_file(path):
    with open(DIR + '/' + path) as fp:
        return xml_nows(fp.read())


def png_file(path):
    with open(DIR + '/' + path, 'rb') as fp:
        return fp.read()


def strlist(ls):
    return ','.join(str(p) for p in ls)


def req(url, **kwargs):
    """Perform a get request to the local server."""

    url = 'http://127.0.0.1/' + url
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


def make_point_features(schema, crs, start_x, start_y, rows=10, cols=5, gap=100):
    """Generate features with on a rectangular grid of points"""

    features = []

    for r in range(rows):
        for c in range(cols):
            uid = r * cols + (c + 1)

            atts = {}

            for k, v in schema.items():
                if v == 'int':
                    atts[k] = uid * 100
                if v == 'float':
                    atts[k] = uid * 200.0
                if v == 'str':
                    atts[k] = k + '_' + str(uid)
                if v == 'date':
                    atts[k] = datetime.datetime(2019, 1, 1) + datetime.timedelta(days=uid - 1)

            shape = {
                'crs': crs,
                'geometry': {
                    'type': 'Point',
                    'coordinates': [start_x + c * gap, start_y + r * gap]
                }
            }

            features.append(gws.gis.feature.new({
                'uid': uid,
                'attributes': atts,
                'shape': shape
            }))

    return features
