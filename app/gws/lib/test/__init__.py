"""Support tests"""

import datetime
import http.cookies
import inspect
import os.path
import shutil
import sys
import time

import psycopg2
import psycopg2.extras
import pytest
import werkzeug.test
import werkzeug.wrappers

import gws
import gws.base.web.web_app
import gws.config
import gws.core.tree
import gws.base.feature
import gws.lib.json2
import gws.lib.net
import gws.lib.os2
import gws.lib.password
import gws.lib.vendor.slon
import gws.gis.mpx.config
import gws.server.control
import gws.spec.runtime

fixture = pytest.fixture

# configuration for tests, see bin/_test.py

CONFIG = {}

TEMP_DIR = '/tmp'

MANIFEST_PATH = TEMP_DIR + '/gws_test_manifest.json'

DEFAULT_MANIFEST = {
    'withStrictConfig': True,
    'withFallbackConfig': False,
}

# GWS configuration defaults

SESSION_STORE_PATH = '/tmp/gws_test_session_store.sqlite'

GWS_CONFIG_PATH = '/gws-var/gws_test_gws_config.json'

GWS_CONFIG_DEFAULTS = {
    'server': {
        'log': {'level': 'DEBUG'},
        'mapproxy': {'forceStart': True},
    },
    'auth': {
        'sessionStore': 'sqlite',
        'sessionStorePath': SESSION_STORE_PATH,
    },
}


# test runner

def main(args):
    CONFIG.update(gws.lib.json2.from_path('/gws-var/TEST_CONFIG.json'))
    gws.lib.json2.to_path(MANIFEST_PATH, CONFIG.get('MANIFEST', DEFAULT_MANIFEST))

    rootdir = gws.APP_DIR + '/gws'
    files = list(gws.lib.os2.find_files(rootdir, r'_test\.py'))
    spec = True

    if args and not args[0].startswith('-'):
        pattern = args.pop(0)
        if pattern.startswith('nospec:'):
            pattern = pattern.split(':')[1]
            spec = False
        if pattern:
            files = [f for f in files if pattern in f]

    if not files:
        gws.log.error(f'no files to test')
        return

    _sort_order = ['/core/', '/lib/', '/base/', '/plugin/']

    def _sort_key(path):
        for n, s in enumerate(_sort_order):
            if s in path:
                return n, path
        return 99, path

    files.sort(key=_sort_key)

    if spec:
        gws.spec.runtime.create_and_store()

    pytest_args = ['-c', CONFIG['PYTEST_INI_PATH'], '--rootdir', rootdir]
    pytest_args.extend(args)
    pytest_args.extend(files)
    gws.log.debug(f'running pytest with args: {pytest_args}')
    pytest.main(pytest_args)


##


def setup():
    gws.log.debug(f'TEST:setup')
    pass


def teardown():
    gws.log.debug(f'TEST:teardown')

    gws.lib.os2.unlink(SESSION_STORE_PATH)

    gws.base.web.web_app.reload()

    gws.core.tree.unregister_ext()
    gws.config.deactivate()

    web_server_command('reset')


##

def configure(config, parse=True):
    def _dct2cfg(d):
        if isinstance(d, dict):
            return gws.Config({k: _dct2cfg(v) for k, v in d.items()})
        if isinstance(d, (list, tuple)):
            return [_dct2cfg(v) for v in d]
        return d

    gws.log.debug(f'TEST:configure')

    if isinstance(config, str):
        config = gws.lib.vendor.slon.parse(config, as_object=True)
    dct = gws.deep_merge(GWS_CONFIG_DEFAULTS, config)

    config = _dct2cfg(dct)
    gws.lib.json2.to_path(GWS_CONFIG_PATH, config, pretty=True)

    if parse:
        r = gws.config.configure(manifest_path=MANIFEST_PATH, config_path=GWS_CONFIG_PATH)
    else:
        r = gws.config.configure(manifest_path=MANIFEST_PATH, config=config)

    gws.config.activate(r)
    gws.config.store(r)

    return r


def configure_and_reload(config, parse=True):
    def _wait_for_port(service):
        while 1:
            port = CONFIG[f'service.gws.{service}_port']
            url = 'http://' + CONFIG['runner.host_name'] + ':' + str(port)
            res = gws.lib.net.http_request(url)
            if res.ok:
                return
            gws.log.debug(f'TEST:waiting for {service}:{port}')
            sleep(2)

    r = configure(config, parse)
    gws.server.control.reload(['mapproxy', 'web'])

    for service in 'http', 'mpx':
        _wait_for_port(service)

    return r


def root():
    return gws.config.root()


# requests and responses

def local_request(url, **kwargs):
    """Perform a get request to the local server."""

    return gws.lib.net.http_request('http://127.0.0.1' + '/' + url, **kwargs)


class ClientCmdResponse(gws.Data):
    status: int
    json: dict
    cookies: dict
    response: werkzeug.wrappers.Response


def client_cmd_request(cmd, params, cookies=None, headers=None) -> ClientCmdResponse:
    gws.log.debug(f'TEST:client_cmd_request {cmd}')

    client = _prepare_client(cookies)

    resp = client.open(
        method='POST',
        path='/_/' + cmd,
        data=gws.lib.json2.to_string({'params': params}),
        content_type='application/json',
        headers=headers,
    )

    js = None
    try:
        js = gws.lib.json2.from_string(resp.data)
    except:
        pass

    cookie_headers = ';'.join(v for k, v in resp.headers if k == 'Set-Cookie')
    response_cookies = {}

    mor: http.cookies.Morsel
    for k, mor in http.cookies.SimpleCookie(cookie_headers).items():
        response_cookies[k] = dict(mor)
        response_cookies[k]['value'] = mor.value

    return ClientCmdResponse(
        status=resp.status_code,
        json=js,
        cookies=response_cookies,
        response=resp,
    )


def _prepare_client(cookies):
    client = werkzeug.test.Client(
        gws.base.web.web_app.application,
        werkzeug.wrappers.BaseResponse)

    if cookies:
        for k, v in cookies.items():
            if not v:
                client.delete_cookie('localhost', k)
            elif isinstance(v, str):
                client.set_cookie('localhost', k, v)
            else:
                client.set_cookie('localhost', k, **v)

    return client


# web server

def web_server_command(cmd, params=None):
    base_url = f"http://{CONFIG['runner.host_name']}:{CONFIG['service.web.port']}"
    params = params or {}
    params['cmd'] = cmd
    res = gws.lib.net.http_request(
        base_url,
        data=gws.lib.json2.to_string(params),
        method='post'
    )
    return gws.lib.json2.from_string(res.text)


def web_server_poke(pattern, response):
    return web_server_command('poke', {'pattern': pattern, 'response': response})


def web_server_begin_capture():
    return web_server_command('begin_capture')


def web_server_end_capture():
    res = web_server_command('end_capture')
    return [gws.lib.net.parse_url('http://host' + u) for u in res['urls']]


def web_server_create_wms(config):
    web_server_command('create_wms', {'config': config})


def web_server_url(url):
    base_url = f"http://{CONFIG['runner.host_name']}:{CONFIG['service.web.port']}"
    return base_url + '/' + url


# features


def make_features(name, geom_type, columns, crs, xy, rows, cols, gap):
    features = []

    sx, sy = xy

    for r in range(rows):
        for c in range(cols):
            uid = r * cols + (c + 1)

            atts = []

            for k, v in columns.items():
                val = ''
                if v == 'int':
                    val = uid * 100
                if v == 'float':
                    val = uid * 200.0
                if v in ('varchar', 'text'):
                    val = f"{name}/{uid}"
                if v == 'date':
                    val = datetime.datetime(2019, 1, 1) + datetime.timedelta(days=uid - 1)
                atts.append(gws.Attribute(name=k, value=val))

            x = sx + c * gap
            y = sy + r * gap

            geom = None

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

            features.append(gws.base.feature.from_props(gws.Data(
                uid=uid,
                attributes=atts,
                shape={'crs': crs, 'geometry': geom} if geom else None
            )))

    return features


def geojson_make_features(path, geom_type, columns, crs, xy, rows, cols, gap):
    name = gws.lib.os2.parse_path(path)['name']
    features = make_features(name, geom_type, columns, crs, xy, rows, cols, gap)
    text = gws.lib.json2.to_pretty_string({
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': crs}},
        'features': [f.to_geojson() for f in features],
    })
    write_file_if_changed(path, text)


# postgres

def postgres_connect_params():
    return {
        'database': CONFIG['service.postgres.database'],
        'user': CONFIG['service.postgres.user'],
        'password': CONFIG['service.postgres.password'],
        'port': CONFIG['service.postgres.port'],
        'host': CONFIG['runner.host_name'],
    }


def postgres_connection():
    return psycopg2.connect(**postgres_connect_params())


def postgres_make_features(name, geom_type, columns, crs, xy, rows, cols, gap):
    colnames = list(columns)
    coldefs = [f'{c} {columns[c]}' for c in colnames]

    features = make_features(name, geom_type, columns, crs, xy, rows, cols, gap)
    shape = features[0].shape
    if shape:
        colnames.append('p_geom')
        coldefs.append(f'p_geom GEOMETRY({shape.geometry_type},{shape.srid})')

    data = []
    for f in features:
        rec = [a.value for a in f.attributes]
        if f.shape:
            rec.append(f.shape.ewkb_hex)
        data.append(rec)

    conn = postgres_connection()
    cur = conn.cursor()

    cur.execute(f'BEGIN')
    cur.execute(f'DROP TABLE IF EXISTS {name}')
    cur.execute(f'''
        CREATE TABLE {name} (
            id SERIAL PRIMARY KEY,
            {','.join(coldefs)}
        )
    ''')
    cur.execute(f'COMMIT')

    cur.execute(f'BEGIN')
    ins = f'''INSERT INTO {name} ({','.join(colnames)}) VALUES %s'''
    psycopg2.extras.execute_values(cur, ins, data)
    cur.execute(f'COMMIT')

    conn.close()


def postgres_drop_table(name):
    conn = postgres_connection()
    cur = conn.cursor()

    cur.execute(f'BEGIN')
    cur.execute(f'DROP TABLE IF EXISTS {name}')
    cur.execute(f'COMMIT')
    conn.close()


# utilities

def make_users_json(lst):
    path = '/tmp/gws_test_users.json'
    if lst is None:
        gws.lib.os2.unlink(path)
        return None

    for v in lst:
        v['password'] = gws.lib.password.encode(v['password'])
    gws.lib.json2.to_path(path, lst)
    return path


def register_ext(class_name, cls):
    gws.core.tree.register_ext(class_name, cls)


def write_file(path, text):
    pp = gws.lib.os2.parse_path(path)
    if pp['dirname'].startswith(TEMP_DIR):
        gws.ensure_dir(pp['dirname'])
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


def write_file_if_changed(path, text):
    curr = read_file(path)
    if text != curr:
        write_file(path, text)


def copy_file(path, dir):
    shutil.copy(path, dir)


def rel_path(path):
    f = inspect.stack(2)[1].filename
    return os.path.join(os.path.dirname(f), path)


def sleep(n):
    time.sleep(n)


def raises(exc):
    return pytest.raises(exc)


def dict_of(x):
    if gws.is_data_object(x):
        # noinspection PyTypeChecker
        return dict(sorted(vars(x).items()))
    return x


# div. geodata

class POINTS:
    # PT Passy
    paris = [254451, 6250716]

    # PT Maxplatz
    dus = [753834, 6660874]

    # Linden x Birken Str
    dus1 = [756871, 6661810]

    # PT Wehrhahn
    dus2 = [756766, 6661801]

    # Linden x Mendelssohn Str
    dus3 = [757149, 6661832]

    # PT Neßler Str.
    dus4 = [765513, 6648529]

    # PT Gärdet
    stockholm = [2014778, 8255502]

    # PT Ustinksy Most
    moscow = [4189555, 7508535]

    # PT Cho Ba Chieu / Gia Dinh
    vietnam = [11877461, 1209716]

    # PT Flemington Racecourse / Melbourne
    australia = [16131032, -4549421]

    # Yarawa Rd x Namara Rd
    fiji = [19865901, -2052085]

    # Main Road Y junction
    pitcairn = [-14482452, -2884039]

    # PT Allende
    mexico = [-11035867, 2206279]

    # Park Av x Carson St
    memphis = [-10014603, 4178550]

    # PT Broadway & West 3rd
    ny = [-8237102, 4972223]

    # PT Lime Str
    liverpool = [-331463, 7058753]

    # PT East India Dock Rd
    london = [-48, 6712663]

    # PT Tema Harbour
    ghana = [201, 627883]
