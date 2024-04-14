"""Test utilities."""

import inspect
import io
import json
import os
import shutil
import time
import re
import configparser

import pytest

import gws
import gws.base.feature
import gws.lib.vendor.slon
import gws.spec.runtime
import gws.base.web.wsgi_app
import gws.base.auth
import gws.config
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.net
import gws.lib.sa as sa
import gws.lib.password
import gws.lib.image as image

fixture = pytest.fixture
raises = pytest.raises

OPTIONS = {}


def option(name, default=None):
    return OPTIONS.get(name, default)


def work_dir():
    return option('runner.work_dir')


##

_SPEC = None
_CONFIG_DEFAULTS = '''
    database.providers+ { type "postgres" serviceName "gws_test_postgres" } 
'''


def gws_root():
    global _SPEC
    if _SPEC is None:
        _SPEC = gws.spec.runtime.create(work_dir() + '/MANIFEST.json', read_cache=False, write_cache=False)
    return gws.u.create_root(_SPEC)


def gws_configure(config: str):
    wd = work_dir()

    config = _CONFIG_DEFAULTS + '\n' + config
    dct = gws.lib.vendor.slon.parse(config, as_object=True)

    gws.lib.jsonx.to_path(f'{wd}/config.json', dct, pretty=True)
    root = gws.config.configure(
        manifest_path=f'{wd}/MANIFEST.json',
        config_path=f'{wd}/config.json',
        with_spec_cache=False
    )
    gws.config.activate(root)
    return root


def gws_system_user():
    return gws.base.auth.user.SystemUser(None, roles=[])


def gws_model_context(**kwargs):
    kwargs.setdefault('op', gws.ModelOperation.read)
    kwargs.setdefault('user', gws_system_user())
    return gws.ModelContext(**kwargs)


##


##

_PG_ENGINE = None
_PG_CONN = None


def pg_connect():
    # kwargs.setdefault('pool_pre_ping', True)
    # kwargs.setdefault('echo', self.root.app.developer_option('db.engine_echo'))

    global _PG_ENGINE, _PG_CONN

    if _PG_ENGINE is None:
        params = {
            'application_name': 'gws_test',
            'service': option('service.postgres.name'),
        }
        url = gws.lib.net.make_url(
            scheme='postgresql',
            hostname='',
            path='',
            params=params,
        )
        _PG_ENGINE = sa.create_engine(url)

    if _PG_CONN is None:
        _PG_CONN = _PG_ENGINE.connect()

    return _PG_CONN


def pg_create(table_name, col_defs):
    conn = pg_connect()
    conn.execute(sa.text(f'DROP TABLE IF EXISTS {table_name} CASCADE'))
    ddl = _comma(f'{k} {v}' for k, v in col_defs.items())
    conn.execute(sa.text(f'CREATE TABLE {table_name} ( {ddl} )'))
    conn.commit()


def pg_insert(table_name, row_dicts):
    conn = pg_connect()
    conn.execute(sa.text(f'TRUNCATE TABLE {table_name}'))
    if row_dicts:
        names = _comma(k for k in row_dicts[0])
        values = _comma(':' + k for k in row_dicts[0])
        ins = sa.text(f'INSERT INTO {table_name} ( {names} ) VALUES( {values} )')
        conn.execute(ins, row_dicts)
    conn.commit()


def pg_rows(sql: str) -> list[tuple]:
    conn = pg_connect()
    return [tuple(r) for r in conn.execute(sa.text(sql))]


def pg_exec(sql: str, **kwargs):
    conn = pg_connect()
    for s in sql.split(';'):
        if s.strip():
            conn.execute(sa.text(s.strip()), kwargs)
    conn.commit()


##

def model(root, name) -> gws.Model:
    return root.get(name)


def db_model(root, name) -> gws.DatabaseModel:
    return root.get(name)


def feature(model, **atts) -> gws.Feature:
    f = gws.base.feature.new(model=model, record=gws.FeatureRecord(attributes=atts))
    f.attributes = atts
    return f


##


def mockserver_add_snippet(text):
    mockserver_invoke('/__add', data=text)


def mockserver_clear():
    mockserver_invoke('/__del', data='')


def mockserver_invoke(url_path, params: dict = None, data: str | bytes | dict = None) -> str | bytes | dict | None:
    host = option('service.mockserver.name')
    args = {}
    if data is not None:
        args['method'] = 'POST'
        if isinstance(data, dict):
            args['data'] = gws.lib.jsonx.to_string(data).encode('utf8')
            args['headers'] = {'content-type': 'application/json'}
        elif isinstance(data, str):
            args['data'] = data.encode('utf8')
        else:
            args['data'] = data
    else:
        args['method'] = 'GET'
        args['params'] = params

    url = gws.lib.net.make_url(
        scheme='http',
        hostname=option('service.mockserver.host'),
        port=option('service.mockserver.port'),
        path=url_path
    )
    res = gws.lib.net.http_request(url, **args)
    if not res.ok:
        return

    ct = res.content_type
    if ct == 'application/octet-stream' or ct.startswith('image/'):
        return res.content
    if ct == 'application/json':
        return gws.lib.jsonx.from_string(res.text)
    return res.text


##


def setup():
    pass


def teardown():
    gws.lib.osx.unlink(glob.SESSION_STORE_PATH)
    gws.base.web.wsgi_app.reload()
    gws.config.deactivate()
    mockserv.command('reset')


def make_users_json(lst):
    path = '/tmp/gws_test_users.json'
    if lst is None:
        gws.lib.osx.unlink(path)
        return None

    for v in lst:
        v['password'] = gws.lib.password.encode(v['password'])
    gws.lib.jsonx.to_path(path, lst)
    return path


def write_file(path, text):
    pp = gws.lib.osx.parse_path(path)
    if pp['dirname'].startswith(gws.c.TMP_DIR):
        gws.u.ensure_dir(pp['dirname'])
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


def dict_of(x):
    if gws.u.is_data_object(x):
        # noinspection PyTypeChecker
        return dict(sorted(vars(x).items()))
    return x


def fxml(s):
    s = s.strip()
    s = re.sub(r'>\s+<', '><', s)
    s = re.sub(r'\s*>\s*', '>', s)
    s = re.sub(r'\s*<\s*', '<', s)
    s = re.sub(r'\s+', ' ', s)
    return s


_comma = ','.join


def image_similarity(a: image.Image | gws.Image, b: image.Image) -> float:
    error = 0
    x, y = a.size()
    for i in range(int(x)):
        for j in range(int(y)):
            a_r, a_g, a_b, a_a = a.getpixel((i, j))
            b_r, b_g, b_b, b_a = b.getpixel((i, j))
            error += (a_r - b_r) ** 2
            error += (a_g - b_g) ** 2
            error += (a_b - b_b) ** 2
            error += (a_a - b_a) ** 2
    return error / (3 * x * y)
