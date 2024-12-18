"""Test utilities."""

import contextlib
import os
import re
import typing
from typing import Optional

import pytest
import requests
import werkzeug.test

import gws
import gws.base.auth
import gws.base.feature
import gws.base.shape
import gws.base.web.wsgi_app
import gws.config
import gws.lib.crs
import gws.lib.cli as cli
import gws.lib.jsonx
import gws.lib.net
import gws.lib.sa as sa
import gws.lib.vendor.slon
import gws.spec.runtime
import gws.test.mock

##

mock = gws.test.mock

fixture = pytest.fixture
raises = pytest.raises

monkey_patch = pytest.MonkeyPatch.context

cast = typing.cast

exec = gws.lib.cli.exec

##

OPTIONS = {}


def option(name, default=None):
    return OPTIONS.get(name, default)


##


def _config_defaults():
    return f'''
        database.providers+ {{ 
            uid "GWS_TEST_POSTGRES_PROVIDER" 
            type "postgres" 
            host     {OPTIONS['service.postgres.host']!r}
            port     {OPTIONS['service.postgres.port']!r}
            username {OPTIONS['service.postgres.user']!r}
            password {OPTIONS['service.postgres.password']!r}
            database {OPTIONS['service.postgres.database']!r}
            schemaCacheLifeTime 0 
        }}
    '''


def _to_data(x):
    if isinstance(x, gws.Data):
        for k, v in vars(x).items():
            setattr(x, k, _to_data(v))
        return x
    if isinstance(x, dict):
        d = gws.Data()
        for k, v in x.items():
            setattr(d, k, _to_data(v))
        return d
    if isinstance(x, list):
        return [_to_data(y) for y in x]
    if isinstance(x, tuple):
        return tuple(_to_data(y) for y in x)
    return x


_GWS_SPEC_DICT = None


def gws_specs() -> gws.SpecRuntime:
    global _GWS_SPEC_DICT

    if _GWS_SPEC_DICT is None:
        base = option('BASE_DIR')
        _GWS_SPEC_DICT = gws.spec.runtime.get_spec(
            f'{base}/config/MANIFEST.json',
            read_cache=False,
            write_cache=False
        )

    return gws.spec.runtime.Object(_GWS_SPEC_DICT)


def gws_root(config: str = '', specs: gws.SpecRuntime = None, activate=True, defaults=True):
    config = config or ''
    if defaults:
        config = _config_defaults() + '\n' + config
    config = f'server.log.level {gws.log.get_level()}\n' + config
    parsed_config = _to_data(gws.lib.vendor.slon.parse(config, as_object=True))
    specs = mock.register(specs or gws_specs())
    root = gws.config.initialize(specs, parsed_config)
    if root.configErrors:
        for err in root.configErrors:
            gws.log.error(f'CONFIGURATION ERROR: {err}')
        raise gws.ConfigurationError('config failed')
    if not activate:
        return root
    root = gws.config.activate(root)
    return root


def gws_system_user():
    return gws.base.auth.user.SystemUser(None, roles=[])


def get_db(root):
    return root.get('GWS_TEST_POSTGRES_PROVIDER')


##

# ref: https://werkzeug.palletsprojects.com/en/3.0.x/test/

class TestResponse(werkzeug.test.TestResponse):
    cookies: dict[str, werkzeug.test.Cookie]


def _wz_request(root, **kwargs):
    client = werkzeug.test.Client(gws.base.web.wsgi_app.make_application(root))

    cookies = cast(list[werkzeug.test.Cookie], kwargs.pop('cookies', []))
    for c in cookies:
        client.set_cookie(
            key=c.key,
            value=c.value,
            max_age=c.max_age,
            expires=c.expires,
            path=c.path,
            domain=c.domain,
            secure=c.secure,
            httponly=c.http_only,
        )

    res = client.open(**kwargs)

    # for some reason, responses do not include cookies, work around this
    res.cookies = {c.key: c for c in (client._cookies or {}).values()}
    return res


class http:
    @classmethod
    def get(cls, root, url, **kwargs) -> TestResponse:
        url = re.sub(r'\s+', '', url.strip())
        url = '/' + url.strip('/')
        return _wz_request(root, method='GET', path=url, **kwargs)

    @classmethod
    def api(cls, root, cmd, request=None, **kwargs) -> TestResponse:
        path = gws.c.SERVER_ENDPOINT
        if cmd:
            path += '/' + cmd
        return _wz_request(root, method='POST', path=path, json=request or {}, **kwargs)


##

class pg:
    saEngine: Optional[sa.Engine] = None
    saConn: Optional[sa.Connection] = None

    @classmethod
    def connect(cls):
        # kwargs.setdefault('pool_pre_ping', True)
        # kwargs.setdefault('echo', self.root.app.developer_option('db.engine_echo'))

        if not cls.saEngine:
            url = gws.lib.net.make_url(
                scheme='postgresql',
                username=OPTIONS['service.postgres.user'],
                password=OPTIONS['service.postgres.password'],
                hostname=OPTIONS['service.postgres.host'],
                port=OPTIONS['service.postgres.port'],
                path=OPTIONS['service.postgres.database'],
            )
            cls.saEngine = sa.create_engine(url)

        if not cls.saConn:
            cls.saConn = cls.saEngine.connect()

        return cls.saConn

    @classmethod
    def close(cls):
        if cls.saConn:
            cls.saConn.close()
            cls.saConn = None

    @classmethod
    def create(cls, table_name: str, col_defs: dict):
        conn = cls.connect()
        conn.execute(sa.text(f'DROP TABLE IF EXISTS {table_name} CASCADE'))
        ddl = _comma(f'{k} {v}' for k, v in col_defs.items())
        conn.execute(sa.text(f'CREATE TABLE {table_name} ( {ddl} )'))
        conn.commit()

    @classmethod
    def clear(cls, table_name: str):
        conn = cls.connect()
        conn.execute(sa.text(f'TRUNCATE TABLE {table_name}'))
        conn.commit()

    @classmethod
    def insert(cls, table_name: str, row_dicts: list[dict]):
        conn = cls.connect()
        conn.execute(sa.text(f'TRUNCATE TABLE {table_name}'))
        if row_dicts:
            names = _comma(k for k in row_dicts[0])
            values = _comma(':' + k for k in row_dicts[0])
            ins = sa.text(f'INSERT INTO {table_name} ( {names} ) VALUES( {values} )')
            conn.execute(ins, row_dicts)
        conn.commit()

    @classmethod
    def rows(cls, sql: str) -> list[tuple]:
        conn = cls.connect()
        return [tuple(r) for r in conn.execute(sa.text(sql))]

    @classmethod
    def content(cls, sql_or_table_name: str) -> list[tuple]:
        if not sql_or_table_name.lower().startswith('select'):
            sql_or_table_name = f'SELECT * FROM {sql_or_table_name}'
        return cls.rows(sql_or_table_name)

    @classmethod
    def exec(cls, sql: str, **kwargs):
        conn = cls.connect()
        for s in sql.split(';'):
            if s.strip():
                conn.execute(sa.text(s.strip()), kwargs)
        conn.commit()


##

class log:
    _buf = []

    @classmethod
    def write(cls, s):
        cls._buf.append(s)

    @classmethod
    def reset(cls):
        cls._buf = []

    @classmethod
    def get(cls):
        r = cls._buf
        cls._buf = []
        return r


##


def feature_from_dict(model, atts) -> gws.Feature:
    f = gws.base.feature.new(model=model, record=gws.FeatureRecord(attributes=atts))
    f.attributes = atts
    return f


def feature(model, **atts) -> gws.Feature:
    return feature_from_dict(model, atts)


def ewkb(wkt: str, srid=3857):
    shape = gws.base.shape.from_wkt(wkt, default_crs=gws.lib.crs.get(srid))
    return shape.to_ewkb()


def model_context(**kwargs):
    kwargs.setdefault('op', gws.ModelOperation.read)
    kwargs.setdefault('user', gws_system_user())
    return gws.ModelContext(**kwargs)


##


class mockserver:
    @classmethod
    def add(cls, text):
        cls.post('__add', data=text)

    @classmethod
    def set(cls, text):
        cls.post('__set', data=text)

    @classmethod
    def reset(cls):
        cls.post('__del')

    @classmethod
    def post(cls, verb, data=''):
        requests.post(cls.url(verb), data=data)

    @classmethod
    def url(cls, path=''):
        h = OPTIONS.get('service.mockserver.host')
        p = OPTIONS.get('service.mockserver.port')
        u = f'http://{h}:{p}'
        if path:
            u += '/' + path
        return u


##

def fxml(s):
    s = re.sub(r'\s+', ' ', s.strip())
    return (
        s
        .replace(' <', '<')
        .replace('< ', '<')
        .replace(' >', '>')
        .replace('> ', '>')
    )


##

def unlink(path):
    if os.path.isfile(path):
        os.unlink(path)
        return
    if os.path.isdir(path):
        for de in os.scandir(path):
            unlink(de.path)
        os.rmdir(path)


def ensure_dir(path, clear=False):
    os.makedirs(path, exist_ok=True)
    if clear:
        for de in os.scandir(path):
            unlink(de.path)


def path_in_base_dir(path):
    base = option('BASE_DIR')
    return f'{base}/{path}'


@contextlib.contextmanager
def temp_file_in_base_dir(content='', keep=False):
    # for exec/chmod tests, which cannot use tmp_path

    base = option('BASE_DIR')
    d = f'{base}/tmp'
    ensure_dir(d)

    p = d + '/' + gws.u.random_string(16)
    gws.u.write_file(p, content)
    yield p

    if not keep:
        unlink(p)


@contextlib.contextmanager
def temp_dir_in_base_dir(keep=False):
    base = option('BASE_DIR')

    d = f'{base}/tmp/' + gws.u.random_string(16)
    ensure_dir(d, clear=True)
    yield d

    if not keep:
        unlink(d)


_comma = ','.join
