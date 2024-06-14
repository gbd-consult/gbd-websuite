"""Test utilities."""

import typing
from typing import Optional
import re

import pytest
import requests

import gws
import gws.base.auth
import gws.base.feature
import gws.config
import gws.lib.cli
import gws.lib.net
import gws.lib.sa as sa
import gws.lib.vendor.slon
import gws.spec.runtime
import gws.test.mocks

##

mocks = gws.test.mocks

fixture = pytest.fixture
raises = pytest.raises

cast = typing.cast

exec = gws.lib.cli.exec

##

OPTIONS = {}


def option(name, default=None):
    return OPTIONS.get(name, default)


def work_dir():
    return option('runner.work_dir')


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
        server.log.level "INFO" 
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
        _GWS_SPEC_DICT = gws.spec.runtime.get_spec(work_dir() + '/MANIFEST.json', read_cache=False, write_cache=False)
    return gws.spec.runtime.Object(_GWS_SPEC_DICT)


def gws_root(config: str = '', specs: gws.SpecRuntime = None):
    print('')
    config = _config_defaults() + '\n' + config
    parsed_config = _to_data(gws.lib.vendor.slon.parse(config, as_object=True))
    specs = mocks.register(specs or gws_specs())
    root = gws.config.initialize(specs, parsed_config)
    return gws.config.activate(root)


def gws_system_user():
    return gws.base.auth.user.SystemUser(None, roles=[])


def gws_model_context(**kwargs):
    kwargs.setdefault('op', gws.ModelOperation.read)
    kwargs.setdefault('user', gws_system_user())
    return gws.ModelContext(**kwargs)


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


def feature(model, **atts) -> gws.Feature:
    f = gws.base.feature.new(model=model, record=gws.FeatureRecord(attributes=atts))
    f.attributes = atts
    return f


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


_comma = ','.join
